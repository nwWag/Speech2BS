//
//  ViewController.swift
//  FaceRecorderSwift
//
//  Created by Timo Menzel on 24.06.21.
//

import UIKit
import SceneKit
import ARKit
import ReplayKit
import Photos

class ViewController: UIViewController, ARSCNViewDelegate, AVCaptureFileOutputRecordingDelegate, AVCaptureDepthDataOutputDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: (any Error)?) {
    }
    

    @IBOutlet var sceneView: ARSCNView!
    
    let recorder = RPScreenRecorder.shared()
    
    @IBOutlet weak var RecordButton: UIButton!
    @IBOutlet weak var renderModeButton: UIButton!
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        sceneView.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        RecordButton.backgroundColor = UIColor.white
        RecordButton.tintColor = UIColor.black
        RecordButton.layer.borderWidth = 3
        RecordButton.layer.borderColor = CGColor(gray: 0, alpha: 1)
        
        renderModeButton.backgroundColor = UIColor.white
        renderModeButton.tintColor = UIColor.black
        renderModeButton.layer.borderWidth = 3
        renderModeButton.layer.borderColor = CGColor(gray: 0, alpha: 1)
        renderModeButton.titleLabel?.lineBreakMode = .byWordWrapping;
        
        guard ARFaceTrackingConfiguration.isSupported else {
            fatalError("Face tracking is not supported on this device!")
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)

        // Add depth output
        guard captureSession.canAddOutput(depthDataOutput) else { fatalError() }
        captureSession.addOutput(depthDataOutput)
        
        if let connection = depthDataOutput.connection(with: .depthData) {
            connection.isEnabled = true
            depthDataOutput.isFilteringEnabled = false
            depthDataOutput.setDelegate(self, callbackQueue: dataOutputQueue)
        } else {
            print("No AVCaptureConnection")
        }
        
        depthCapture.prepareForRecording()
        captureSession.startRunning()
        
        // Create a session configuration
        let configuration = ARFaceTrackingConfiguration()

        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    var recording = false
    var assetWriter: AVAssetWriter!
    var videoInput: AVAssetWriterInput!
    var audioMicInput: AVAssetWriterInput!
    var timer = Timer()
    var blendshapeWeights = Array(repeating: [Float](), count: 68) // timestamp + 52 blendshape weights + 4 head quaternion values + 4 left eye quaternion values + 4 right eye quaternion values + 3 lookAtPoint
    var currentBlendshapes: [ARFaceAnchor.BlendShapeLocation : NSNumber] = [:]
    var currentGeom: Array<simd_float3> = Array()
    var geoms = [[simd_float3]]()
    var currentLeftEye: simd_float4x4!
    var currentRightEye: simd_float4x4!
    var currentTransform: simd_float4x4!
    var currentLookAtPoint: simd_float3!
    var count = 0
    var captureWeights = false
    var startTime: CFTimeInterval = 0
    
    var prepareMappings = true
    var Index2BSLocation: [Int : ARFaceAnchor.BlendShapeLocation] = [:]
    var BSLocation2Name: [ARFaceAnchor.BlendShapeLocation : String] = [:]
    
    var writer = 0
    var isWriting = false
    
    let captureSession = AVCaptureSession()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    private let dataOutputQueue = DispatchQueue(label: "dataOutputQueue")
    private let depthCapture = DepthCapture()
    
    @objc func saveBlendshapeWeights()
    {
        if(recording && captureWeights && !isWriting)
        {
            isWriting = true
            writer = writer + 1
            print("Writer: \(writer) ")
            print(currentBlendshapes.count)
            let blendshapes = currentBlendshapes
            
            // head transform
            let transform = currentTransform!
            let head_quaternion: simd_quatf = simd_quatf(transform)
            
            // left eye
            let leftEye = currentLeftEye!
//            leftEye[3][0] = 0
//            leftEye[3][1] = 0
//            leftEye[3][2] = 0
            let left_eye_quaternion: simd_quatf = simd_quatf(leftEye)
            
            // right eye
            let rightEye = currentLeftEye!
//            rightEye[3][0] = 0
//            rightEye[3][1] = 0
//            rightEye[3][2] = 0
            let right_eye_quaternion: simd_quatf = simd_quatf(rightEye)
            
            let lookAtPoint = currentLookAtPoint!
            
            for (idx, _) in blendshapeWeights.enumerated()
            {
                if idx == 0 { // 1 line for timestamp
                    let now = CACurrentMediaTime()
                    blendshapeWeights[idx].append(Float(now - startTime))
                }
                else if idx < 53 // 52 lines for blendshape weights
                {
                    let blendshapeLocation = Index2BSLocation[idx-1]!
                    blendshapeWeights[idx].append(Float(truncating: blendshapes[blendshapeLocation]!))
                }
                else if idx < 57
                {
                    blendshapeWeights[idx].append(head_quaternion.vector[idx - 53])
                }
                else if idx < 61
                {
                    blendshapeWeights[idx].append(left_eye_quaternion.vector[idx - 57])
                }
                else if idx < 65
                {
                    blendshapeWeights[idx].append(right_eye_quaternion.vector[idx - 61])
                }
                else
                {
                    blendshapeWeights[idx].append(lookAtPoint[idx - 65])
                }
            }
            geoms.append(currentGeom)
            print("Writer: \(writer) ")
            count += 1
            isWriting = false
        }
    }
    
    func writeBlendshapeWeightsToFile(){
        let filename = current_path.appendingPathComponent("weights.csv")
        let filename_geom = current_path.appendingPathComponent("geom.csv")
        var data = ""
        var data_geom = ""
        // frame number
        for i in 0...blendshapeWeights[0].count
        {
            data.append(",\(i)")
        }
        data.append("\n")
        
        // blendshapeWeights
        for i in 0...blendshapeWeights.count-1 {
            for j in 0...blendshapeWeights[i].count-1
            {
                if(j == 0) {
                    if(i == 0) {
                        data.append("timestamp")
                    }
                    else if (i < 53) {
                        data.append("\(BSLocation2Name[Index2BSLocation[i-1]!]!)")
                    }
                    else if (i == 53) {
                        data.append("head x")
                    }
                    else if (i == 54) {
                        data.append("head y")
                    }
                    else if (i == 55) {
                        data.append("head z")
                    }
                    else if (i == 56) {
                        data.append("head w")
                    }
                    else if (i == 57) {
                        data.append("left x")
                    }
                    else if (i == 58) {
                        data.append("left y")
                    }
                    else if (i == 59) {
                        data.append("left z")
                    }
                    else if (i == 60) {
                        data.append("left w")
                    }
                    else if (i == 61) {
                        data.append("right x")
                    }
                    else if (i == 62) {
                        data.append("right y")
                    }
                    else if (i == 63) {
                        data.append("right z")
                    }
                    else if (i == 64) {
                        data.append("right w")
                    }
                    else if (i == 65) {
                        data.append("lookAtPoint x")
                    }
                    else if (i == 66) {
                        data.append("lookAtPoint y")
                    }
                    else if (i == 67) {
                        data.append("lookAtPoint z")
                    }
                }
                else
                {
                    data.append(",\(blendshapeWeights[i][j-1])")
                }
            }
            data.append("\n")
        }
        
        // write dataOBJ
        do {
            try data.write(to: filename, atomically: false, encoding: .utf8)
        }
        catch{
            print(error)
            fatalError("Writing error while writing OBJ file!")
            
        }
        
        
        for v_id in 0...geoms[0].count-1 {
            for f_id in 0...geoms.count-1{
  
                    data_geom.append(" \(geoms[f_id][v_id].x) \(geoms[f_id][v_id].y) \(geoms[f_id][v_id].z)")
                
            }
            data_geom.append("\n")
        }

        // write dataOBJ
        do {
            try data_geom.write(to: filename_geom, atomically: false, encoding: .utf8)
        }
        catch{
            print(error)
            fatalError("Writing error while writing Geom file!")
            
        }
        
        
        blendshapeWeights = Array(repeating: [Float](), count: 68)
        geoms = [[simd_float3]]()
    }
        
    func createSavingLocation() -> URL {
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            var recordings_folder = dir.appendingPathComponent("ARKit_Recordings")
            //check if an "EBFR_Input" folder is already available
            if !FileManager.default.fileExists(atPath: recordings_folder.absoluteString) {
                do {
                    try FileManager.default.createDirectory(at: recordings_folder, withIntermediateDirectories: true, attributes: nil)
                    } catch {
                        print(error.localizedDescription)
                }
            }
            
            let date = Date()
            let format = DateFormatter()
            format.dateFormat = "yyyy-MM-dd HH:mm:ss"
            let timestamp = format.string(from: date)
            
            recordings_folder = recordings_folder.appendingPathComponent("\(timestamp)")
            do {
            try FileManager.default.createDirectory(at: recordings_folder, withIntermediateDirectories: true, attributes: nil)
            } catch {
                print("Could not create the directory with name \(timestamp).")
                print(error.localizedDescription)
            }
            return recordings_folder
        }
        
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }
    
    var current_path: URL!
    var video_path : URL!
    var startRecording = false
    
    

    @IBAction func onRecordButtonClick(_ sender: Any) {
        if(!recording)
        {
            startRecording = true
            recordingTrafo = simd_float4x4.init(diagonal: simd_float4(repeating: 1))
            //prepare video
            current_path = createSavingLocation()
            video_path = current_path.appendingPathComponent("video.mp4")
            do{
                try self.assetWriter = AVAssetWriter(outputURL: video_path, fileType: .mp4)
            } catch {}
            
            let videoSettings: [String : Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: view.bounds.width,
                AVVideoHeightKey: view.bounds.height
            ]

            videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            videoInput.expectsMediaDataInRealTime = true
            if assetWriter.canAdd(videoInput) {
                assetWriter.add(videoInput)
            }
            
            let audioSettings: [String:Any] = [AVFormatIDKey : kAudioFormatMPEG4AAC,
                AVNumberOfChannelsKey : 2,
                AVSampleRateKey : 44100.0,
                AVEncoderBitRateKey: 192000
            ]

            audioMicInput = AVAssetWriterInput(mediaType: AVMediaType.audio, outputSettings: audioSettings)
            audioMicInput.expectsMediaDataInRealTime = true
            assetWriter.add(audioMicInput)

            RecordButton.setTitle("Stop Recording", for: UIControl.State.normal)
            recorder.isMicrophoneEnabled = true
            
            self.startTime = CACurrentMediaTime()
            self.timer = Timer.scheduledTimer(timeInterval: 1.0/60.0, target: self, selector: #selector(self.saveBlendshapeWeights), userInfo: nil, repeats: true)
            recorder.startCapture(handler: { (cmSampleBuffer, rpSampleBufferType, err) in
                
                if(err != nil) {return}
                
                self.recording = true
                if CMSampleBufferDataIsReady(cmSampleBuffer) {

                    DispatchQueue.main.async {
                        self.captureWeights = true

                        switch rpSampleBufferType {
                        case .video:

//                            print("writing sample....")

                            if self.assetWriter?.status == AVAssetWriter.Status.unknown {

//                                print("Started writing")
                                self.assetWriter?.startWriting()
                                self.assetWriter?.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(cmSampleBuffer))
                            }

                            if self.assetWriter.status == AVAssetWriter.Status.failed {
                                print("StartCapture Error Occurred, Status = \(self.assetWriter.status.rawValue), \(self.assetWriter.error!.localizedDescription) \(self.assetWriter.error.debugDescription)")
                                 return
                            }

                            if self.assetWriter.status == AVAssetWriter.Status.writing {
                                if self.videoInput.isReadyForMoreMediaData {
//                                    print("Writing a sample")
                                    if self.videoInput.append(cmSampleBuffer) == false {
                                         print("problem writing video")
                                    }
                                 }
                             }
                            break

                        case .audioMic:
                            if self.audioMicInput.isReadyForMoreMediaData {
//                                print("audioMic data added")
                                self.audioMicInput.append(cmSampleBuffer)
                            }
                            break

                        default:
                                print(" ")
//                            print("\(rpSampleBufferType)")
//                            print("not a video sample")
                        }
                    }
                }
            }
        )}
        else
        {
            RecordButton.setTitle("Start Recording", for: UIControl.State.normal)
            recorder.stopCapture { (error) in
                if let _ = error { return }
                self.recording = false
                self.captureWeights = false

                guard let videoInput = self.videoInput else { return }
                guard let audioMicInput = self.audioMicInput else { return }
                guard let assetWriter = self.assetWriter else { return }
                guard let videoURL = self.video_path else { return }

                self.timer.invalidate()
                videoInput.markAsFinished()
                audioMicInput.markAsFinished()
                
                
                assetWriter.finishWriting(completionHandler: {
                    PHPhotoLibrary.shared().performChanges({
                            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: videoURL)
                        }) { (saved, error) in

                            if let error = error {
                                print("PHAssetChangeRequest Video Error: \(error.localizedDescription)")
                                return
                            }

                            if saved {
                                // ... show success message
                            }
                        }
                })
                self.count = 0
                self.writeBlendshapeWeightsToFile()
            }
        
        }
        recording = !recording
    }
    
    func prepareBSLocation2Name()
    {
        BSLocation2Name[.browDownLeft] = "BrowDownLeft"
        BSLocation2Name[.browDownRight] = "BrowDownRight"
        BSLocation2Name[.browInnerUp] = "BrowInnerUp"
        BSLocation2Name[.browOuterUpLeft] = "BrowOuterUpLeft"
        BSLocation2Name[.browOuterUpRight] = "BrowOuterUpRight"
        BSLocation2Name[.cheekPuff] = "CheekPuff"
        BSLocation2Name[.cheekSquintLeft] = "CheekSquintLeft"
        BSLocation2Name[.cheekSquintRight] = "CheekSquintRight"
        BSLocation2Name[.eyeBlinkLeft] = "EyeBlinkLeft"
        BSLocation2Name[.eyeBlinkRight] = "EyeBlinkRight"
        BSLocation2Name[.eyeLookDownLeft] = "EyeLookDownLeft"
        BSLocation2Name[.eyeLookDownRight] = "EyeLookDownRight"
        BSLocation2Name[.eyeLookInLeft] = "EyeLookInLeft"
        BSLocation2Name[.eyeLookInRight] = "EyeLookInRight"
        BSLocation2Name[.eyeLookOutLeft] = "EyeLookOutLeft"
        BSLocation2Name[.eyeLookOutRight] = "EyeLookOutRight"
        BSLocation2Name[.eyeLookUpLeft] = "EyeLookUpLeft"
        BSLocation2Name[.eyeLookUpRight] = "EyeLookUpRight"
        BSLocation2Name[.eyeSquintLeft] = "EyeSquintLeft"
        BSLocation2Name[.eyeSquintRight] = "EyeSquintRight"
        BSLocation2Name[.eyeWideLeft] = "EyeWideLeft"
        BSLocation2Name[.eyeWideRight] = "EyeWideRight"
        BSLocation2Name[.jawForward] = "JawForward"
        BSLocation2Name[.jawLeft] = "JawLeft"
        BSLocation2Name[.jawOpen] = "JawOpen"
        BSLocation2Name[.jawRight] = "JawRight"
        BSLocation2Name[.mouthClose] = "MouthClose"
        BSLocation2Name[.mouthDimpleLeft] = "MouthDimpleLeft"
        BSLocation2Name[.mouthDimpleRight] = "MouthDimpleRight"
        BSLocation2Name[.mouthFrownLeft] = "MouthFrownLeft"
        BSLocation2Name[.mouthFrownRight] = "MouthFrownRight"
        BSLocation2Name[.mouthFunnel] = "MouthFunnel"
        BSLocation2Name[.mouthLeft] = "MouthLeft"
        BSLocation2Name[.mouthLowerDownLeft] = "MouthLowerDownLeft"
        BSLocation2Name[.mouthLowerDownRight] = "MouthLowerDownRight"
        BSLocation2Name[.mouthPressLeft] = "MouthPressLeft"
        BSLocation2Name[.mouthPressRight] = "MouthPressRight"
        BSLocation2Name[.mouthPucker] = "MouthPucker"
        BSLocation2Name[.mouthRight] = "MouthRight"
        BSLocation2Name[.mouthRollLower] = "MouthRollLower"
        BSLocation2Name[.mouthRollUpper] = "MouthRollUpper"
        BSLocation2Name[.mouthShrugLower] = "MouthShrugLower"
        BSLocation2Name[.mouthShrugUpper] = "MouthShrugUpper"
        BSLocation2Name[.mouthSmileLeft] = "MouthSmileLeft"
        BSLocation2Name[.mouthSmileRight] = "MouthSmileRight"
        BSLocation2Name[.mouthStretchLeft] = "MouthStretchLeft"
        BSLocation2Name[.mouthStretchRight] = "MouthStretchRight"
        BSLocation2Name[.mouthUpperUpLeft] = "MouthUpperUpLeft"
        BSLocation2Name[.mouthUpperUpRight] = "MouthUpperUpRight"
        BSLocation2Name[.noseSneerLeft] = "NoseSneerLeft"
        BSLocation2Name[.noseSneerRight] = "NoseSneerRight"
        BSLocation2Name[.tongueOut] = "TongueOut"
            
    }
    
    
    var renderMode = 0
    @IBAction func onRenderModeButtonClick(_ sender: Any) {
        renderMode = (renderMode + 1) < 3 ? renderMode + 1 : 0
        print(renderMode)
    }
    
    
    
    // MARK: - ARSCNViewDelegate
    

    // Override to create and configure nodes for anchors added to the view's session.
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        let faceMesh = ARSCNFaceGeometry(device: sceneView.device!)

        let material = faceMesh!.firstMaterial!
        switch renderMode {
        case 0:
            material.fillMode = .lines
            material.lightingModel = .constant
            material.transparency = 1
        case 1:
            material.fillMode = .fill
            material.lightingModel = .physicallyBased
            material.transparency = 1
        case 2:
            material.transparency = 0
            material.lightingModel = .constant
        default:
            material.fillMode = .lines
            material.lightingModel = .constant
            material.transparency = 1
        }

        let node = SCNNode(geometry: faceMesh)
        return node
    }
    
    var recordingTrafo: simd_float4x4 = simd_float4x4(1)
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        
        let material = node.geometry?.firstMaterial
        switch renderMode {
        case 0:
            material?.fillMode = .lines
            material?.lightingModel = .constant
            material?.transparency = 1
        case 1:
            material?.fillMode = .fill
            material?.lightingModel = .physicallyBased
            material?.transparency = 1
        case 2:
            material?.lightingModel = .constant
            material?.transparency = 0
        default:
            material?.fillMode = .lines
            material?.lightingModel = .constant
            material?.transparency = 1
        }

        node.geometry?.firstMaterial = material
        
        if let faceAnchor = anchor as? ARFaceAnchor, let faceGeometry = node.geometry as? ARSCNFaceGeometry {
            if(startRecording) {
                startRecording = false
                recordingTrafo = faceAnchor.transform.inverse
                print(recordingTrafo)
                print("startRecording set")
            }
            currentBlendshapes = faceAnchor.blendShapes
            currentLeftEye = faceAnchor.leftEyeTransform
            currentRightEye = faceAnchor.rightEyeTransform
            currentTransform = recordingTrafo * faceAnchor.transform
            currentLookAtPoint = faceAnchor.lookAtPoint
            faceGeometry.update(from: faceAnchor.geometry)
            currentGeom = faceAnchor.geometry.vertices
            if(prepareMappings)
            {
                prepareMappings = false
                var idx = 0
                for (bsLocation, _) in faceAnchor.blendShapes {
                    Index2BSLocation[idx] = bsLocation;
                    idx += 1
                }
                prepareBSLocation2Name()
            }
        }
    }

    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user
        
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay
        
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
        
    }
}
