import os
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the URLs for the datasets
url_A = "https://physionet.org/content/challenge-2019/1.0.0/training/training_setA"
url_B = "https://physionet.org/content/challenge-2019/1.0.0/training/training_setB"

# Create a directory to save the files
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/A", exist_ok=True)
os.makedirs(f"{save_dir}/B", exist_ok=True)


# Function to download a single .psv file
def download_file(file_url, file_path):
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            content = response.text

            # Find the PSV data within the HTML content
            psv_data_match = re.search(
                r'<pre class="plain"><code>(.*?)</code></pre>', content, re.DOTALL
            )

            if not psv_data_match:
                raise ValueError("PSV data not found in the provided HTML content.")

            # Extract the PSV data
            psv_data = psv_data_match.group(1).strip()

            # Save the extracted PSV data to a file
            with open(file_path, "w") as file:
                file.write(psv_data)

            print(f"Successfully downloaded and saved {file_path}")
        else:
            raise ValueError(
                f"Failed to download {file_path} with status code {response.status_code}"
            )
    except Exception as e:
        raise ValueError(f"Exception occurred while downloading {file_path}: {e}")


# Function to get all .psv file links from a given URL
def get_psv_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    file_links = soup.find_all("a", href=True)
    psv_links = [link["href"] for link in file_links if link["href"].endswith(".psv")]
    return psv_links


# Function to download and save .psv files from a given URL in parallel
def download_psv_files(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    psv_links = get_psv_links(url)
    file_urls = [f"{url}/{link}?download" for link in psv_links]
    file_paths = [os.path.join(save_dir, os.path.basename(link)) for link in psv_links]
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_file, file_url, file_path)
            for file_url, file_path in zip(file_urls, file_paths)
        ]
        for future in as_completed(futures):
            future.result()  # To raise exceptions if any


# Download files from both datasets in parallel
download_psv_files(url_A, f"{save_dir}/A")
download_psv_files(url_B, f"{save_dir}/B")

print("Download completed.")

# Define the directory where the .psv files are saved
data_dir = "data"

# Get the list of all .psv files
psv_files = []
for root, dirs, files in os.walk(data_dir):
    print(f"Found {len(files)} files in {root}")
    psv_files += [os.path.join(root, file) for file in files if file.endswith(".psv")]

# Read all .psv files into a single DataFrame
dfs = []
for file in psv_files:
    try:
        df = pd.read_csv(file, sep="|", engine="python", encoding="utf-8")
        text = os.path.splitext(file)[0]
        patient_id = text.split("\\")[-2] + "_" + text.split("\\")[-1]
        df["patient_id"] = patient_id
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        print(f"Skipping {file}...")


df = pd.concat(dfs, ignore_index=True)
df.to_csv(f"{save_dir}/df_raw_combined.csv", index=False)
