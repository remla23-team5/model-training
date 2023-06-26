"""Module responsible for downloading the raw dataset from google drive"""

import requests
import click


def download_file_from_google_drive(file_id, destination):
    """Downloads the raw dataset from google drive"""
    download_url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(download_url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(download_url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """Utility function for getting the token"""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    """Function to save the response content to the destination file"""
    chunk_size = 32768

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)


@click.command()
@click.argument(
    "file_id", type=click.STRING, default="1bCFMWa1lgymQtj6vukXTrtfF47TeKQLu"
)
@click.argument(
    "destination_path",
    type=click.Path(),
    default="./data/raw/a1_RestaurantReviews_HistoricDump.tsv",
)
def main(file_id, destination_path):
    """Main function to be run with the CLI."""
    download_file_from_google_drive(file_id, destination_path)
    print("Successfully downloaded data from gdrive")


if __name__ == "__main__":
    # Ignore pylint error for click decorated methods
    # pylint: disable=no-value-for-parameter
    main()
