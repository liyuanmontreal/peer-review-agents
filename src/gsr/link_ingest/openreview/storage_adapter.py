import sqlite3
from gsr.data_collection.storage import save_json, save_to_db


def save_forum_bundle(paper_dict: dict, conn: sqlite3.Connection):
    venue_data = {
        "venue_id": paper_dict.get("venue_id") or "unknown",
        "papers": [paper_dict],
    }

    save_json(venue_data)
    save_to_db(venue_data, conn)
