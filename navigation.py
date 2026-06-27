'''
Functions for navigating directories etc.
'''

import sys
import os
import json
import numpy as np
from concurrent.futures import as_completed, ThreadPoolExecutor
import queue
import threading
import subprocess
from tqdm import tqdm
import natsort
import matplotlib.pyplot as plt

def make_gif_from_images(image_prefix, folder, image_filetype='png', output_file='out.gif', framerate=10, output_width=800):
    """
    Create a GIF from a series of images
    
    :param image_prefix: Prefix of the image files (e.g., 'spot_frame_')
    :param folder: Folder containing the image files
    :param image_filetype: File type of the images (default is 'png')
    :param output_file: Name of the output GIF file (default is 'out.gif')
    :param framerate: Frame rate for the GIF (default is 10)
    :param output_width: Output width in pixels for the GIF (default is 800)
    """

    # find all files with the given prefix and filetype

    files = [f for f in os.listdir(folder) if f.startswith(image_prefix) and f.endswith(f'.{image_filetype}')]
    files = natsort.natsorted(files)

    # create a temporary text file listing the images
    list_file = os.path.join(folder, 'frame_list.txt')
    with open(list_file, 'w', encoding='utf-8', newline='\n') as f:
        for file in files:
            f.write(f"file '{file}'\n")
            f.write(f'duration {1/framerate}\n')
        if files:
            f.write(f"file '{files[-1]}'\n")
    
    # create gif using ffmpeg palette workflow for better color quality
    output_path = os.path.join(folder, output_file)
    palette_path = os.path.join(folder, '_gif_palette.png')

    palette_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-vf', 'fps={},scale={}:-1:flags=lanczos,palettegen=stats_mode=diff'.format(framerate, output_width),
        palette_path,
    ]

    gif_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-i', palette_path,
        '-filter_complex', '[0:v]fps={},scale={}:-1:flags=lanczos,format=rgb24[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=2'.format(framerate, output_width),
        '-loop', '0',
        output_path,
    ]

    subprocess.run(
        palette_cmd,
        check=True,
    )

    try:
        subprocess.run(gif_cmd, check=True)
    finally:
        if os.path.exists(palette_path):
            os.remove(palette_path)

def make_gif_of_plot(plot_func_loop, which_to_plot, folder, output_file='out.gif', framerate=10, output_width=800, **kwargs):
    """
    Create a GIF by saving plots to a folder, then running make_gif_from_images on that folder.
    
    :param plot_func: Function that takes a filename and creates a plot
    :param which_to_plot: List of indices for frames to create
    :param folder: Folder to save to and read from
    :param output_file: Name of the output GIF file (default is 'out.gif')
    :param framerate: Frame rate for the GIF (default is 10)
    :param output_width: Output width in pixels for the GIF (default is 800)
    **kwargs: Additional keyword arguments passed to the plotting function
    """

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in which_to_plot:
        filename = os.path.join(folder, f'frame_{i:03d}.png')
        plot_func_loop(i, **kwargs)
        plt.savefig(filename,)
        plt.close()
    
    make_gif_from_images('frame_', folder=folder, output_file=output_file, framerate=framerate, output_width=output_width)

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

def smart_return(*values):
    """Return single value if only one, else return tuple"""
    return values[0] if len(values) == 1 else values

def set_working_directory():
    """
    Set the working directory to the directory of the script this function is run in.
    """

    if hasattr(sys, 'argv') and sys.argv[0]:
        script_path = os.path.abspath(sys.argv[0])
        script_dir = os.path.dirname(script_path)
        os.chdir(script_dir)
        print(f"Working directory set to: {script_dir}")
    else:
        print("Could not determine script path.")
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # print(f"Working directory set to: {os.getcwd()}")

def return_all_files(folder, extension='.npy'):
    """
    Return all files in the specified folder with the given extension.
    
    :param folder: The folder to search in.
    :param extension: The file extension to filter by (default is '.npy').
    :return: List of files with the specified extension.
    """

    if extension is not None:
        return natsort.natsorted([f for f in os.listdir(folder) if f.endswith(extension)])
    else:
        return natsort.natsorted([f for f in os.listdir(folder)])

def select_file(folder, files = None, extension='.npy'):

    if files is None:
        files = return_all_files(folder, extension=extension)
        if len(files) == 0:
            raise FileNotFoundError(f"No {files[0].split('.')[-1]} files found in folder: {folder}")
        else:
            print("Multiple files found in folder. Please specify which file to use.")
            for i, f in enumerate(files):
                print(f"{i}: {f}")
            choice = int(input("Enter the number of the file to use: "))
            filename = os.path.join(folder, files[choice])
    else:
        files = f'{files}{extension}'
        filename = os.path.join(folder, files)

    print(f"Selected file: {filename}")
    return filename

# ==== THREADING
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from tqdm import tqdm

def thread_function(func, args_list, max_workers=None, show_progress=True, 
                   return_results=True, timeout=None, **kwargs):
    """
    Thread an arbitrary function with multiple argument sets.
    
    Parameters:
    -----------
    func : callable
        The function to be threaded
    args_list : list
        List of arguments for each function call. Each element can be:
        - tuple: positional arguments
        - dict: keyword arguments  
        - single value: single positional argument
    max_workers : int, optional
        Maximum number of threads (default: min(32, cpu_count + 4))
    show_progress : bool
        Whether to show progress bar
    return_results : bool
        Whether to collect and return results
    timeout : float, optional
        Timeout for each function call
    **kwargs : dict
        Additional keyword arguments passed to all function calls
    
    Returns:
    --------
    results : dict or None
        Dictionary mapping index -> result if return_results=True
    """
    
    if max_workers is None:
        import os
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    total_tasks = len(args_list)
    results = {} if return_results else None
    
    # Progress tracking
    if show_progress:
        progress_queue = queue.Queue()
        progress_thread = threading.Thread(
            target=_update_progress, 
            args=(progress_queue, total_tasks),
            daemon=True
        )
        progress_thread.start()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        
        for i, args in enumerate(args_list):
            # Handle different argument types
            if isinstance(args, dict):
                # Merge with additional kwargs
                combined_kwargs = {**args, **kwargs}
                future = executor.submit(func, **combined_kwargs)
            elif isinstance(args, (list, tuple)):
                future = executor.submit(func, *args, **kwargs)
            else:
                # Single argument
                future = executor.submit(func, args, **kwargs)
            
            future_to_index[future] = i
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_index, timeout=timeout):
            index = future_to_index[future]
            
            try:
                result = future.result()
                if return_results:
                    results[index] = result
            except Exception as exc:
                print(f"Task {index} generated an exception: {exc}")
                if return_results:
                    results[index] = f"Error: {exc}"
            
            # Update progress
            if show_progress:
                progress_queue.put(1)
            completed += 1
    
    # Wait for progress thread to finish
    if show_progress:
        progress_thread.join(timeout=5)
    
    return results

def _update_progress(progress_queue, total_tasks):
    """Update progress bar in separate thread"""
    completed = 0
    pbar = tqdm(total=total_tasks, desc="Processing")
    
    while completed < total_tasks:
        try:
            progress_queue.get(timeout=1)
            completed += 1
            pbar.update(1)
        except queue.Empty:
            continue
    
    pbar.close()