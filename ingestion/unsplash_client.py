import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env
load_dotenv()


class UnsplashClient:
    """
    Small client to interact with Unsplash API
    """

    BASE_URL = "https://api.unsplash.com"

    def __init__(self, access_key: Optional[str] = None):
        self.access_key = access_key or os.getenv("UNSPLASH_ACCESS_KEY")

        if not self.access_key:
            raise ValueError(
                "Unsplash access key not found. "
                "Set UNSPLASH_ACCESS_KEY in environment variables."
            )

        self.headers = {
            "Authorization": f"Client-ID {self.access_key}"
        }

    def search_photos(
        self,
        query: str,
        page: int = 1,
        per_page: int = 10
    ) -> Dict[str, Any]:
        """
        Search photos on Unsplash
        """

        url = f"{self.BASE_URL}/search/photos"

        params = {
            "query": query,
            "page": page,
            "per_page": per_page
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code != 200:
            raise RuntimeError(
                f"Unsplash API error {response.status_code}: {response.text}"
            )

        return response.json()
