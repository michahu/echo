import os


class DemoSettings:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.port = os.environ.get("API_PORT", 8000)

    def get_base_url(self):
        return f"http://localhost:{self.port}"

    def get_connect_url(self, user_id: str):
        return f"ws://localhost:{self.port}/ws/{user_id}"

    def get_clear_url(self):
        return f"{self.get_base_url()}/clear"

    def get_init_url(self):
        return f"{self.get_base_url()}/init"

    def get_save_url(self):
        return f"{self.get_base_url()}/save"

    def get_save_url_with_custom_folder(self, folder_path: str, datum_id: str):
        return f"{self.get_save_url()}?folder_path={folder_path}&datum_id={datum_id}"


demo_settings = DemoSettings()
