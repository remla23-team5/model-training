import requests
import click


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params={'id': id, 'confirm': token}
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
            if chunk:
                f.write(chunk)


@click.command()
@click.argument("file_id", type=click.STRING, default="1bCFMWa1lgymQtj6vukXTrtfF47TeKQLu")
@click.argument("destination_path", type=click.Path(), default="./data/raw/a1_RestaurantReviews_HistoricDump.tsv")
def main(file_id, destination_path):
    download_file_from_google_drive(file_id, destination_path)
    print("Successfully downloaded data from gdrive")


if __name__ == "__main__":
    main()

