import os
import shutil

def create_folders(args):
    """Check if training and testing folder exists, add output folders if possible and not existing.
       Otherwise, create the output folder."""
    output_folders = []
    for folder_path in [args.train_folder, args.test_folder]:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist.")
        
        folder_name = os.path.basename(folder_path)
        output_folder_path = os.path.join(args.output_folder, folder_name)
        if not os.path.exists(output_folder_path) or args.recreate:
            if args.recreate:
                shutil.rmtree(output_folder_path, ignore_errors=True)
            os.makedirs(output_folder_path)
        
        output_folders.append(output_folder_path)

    return args.train_folder, output_folders[0], args.test_folder, output_folders[1]