import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


print('Downloading CAM-Net_x16_Super_Resolution "Butterfly" model')
download_file_from_google_drive('1vpJtX_E-NuoG4sMWpuQGsWMX18TZlInP',
                                '/path/to/store/CAM_Net_x16_Super_Resolution_Butterfly.pth')

print('Downloading CAM-Net_x2_Decompression model')
download_file_from_google_drive('1g2kkjWa1rdCj36NAb-2S1-fvIcEPUZwG',
                                '/path/to/store/experiments/CAM_Net_x2_Decompression.pth')
