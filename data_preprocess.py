import os
import shutil

import os
import shutil

con = 0
file_names = set()  # Create an empty set to store file names

# Root directory
root_dir = r'C:\Users\sean8_q3zpzqf\Desktop\Deepfake Audio Detection\dataset\FakeAVCeleb_v1.2\RealVideo-FakeAudio'
audio_dir = r'C:\Users\sean8_q3zpzqf\Desktop\Deepfake Audio Detection\dataset\FakeAVCeleb_v1.2\audio\fake'

# Traverse the directory structure
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file_name in filenames:
        if file_name.endswith('.mp4'):
            source_file = os.path.join(dirpath, file_name)
            
            # Generate a new unique file name
            new_file_name = file_name
            counter = 1
            while new_file_name in file_names:
                base_name, extension = os.path.splitext(file_name)
                new_file_name = f"{base_name}_{counter}{extension}"
                counter += 1
            
            destination_file = os.path.join(audio_dir, new_file_name)
            
            shutil.copy(source_file, destination_file)
            print(f'Copied: {file_name} -> {new_file_name}')
            con += 1

            # Add the new file name to the set
            file_names.add(new_file_name)

print(f'Total files copied: {con}')

# read file
# rootfile = os.listdir(r'C:\Users\sean8_q3zpzqf\Desktop\Deepfake Audio Detection\dataset\FakeAVCeleb_v1.2\audio\real')

# for file1 in rootfile:
#     if file1.endswith('.txt'):
#         # remove file
#         os.remove(r'C:\Users\sean8_q3zpzqf\Desktop\Deepfake Audio Detection\dataset\FakeAVCeleb_v1.2\audio\real' + '\\' + file1)