import os
import json

def sort_key(filename):
    """
    Estrae i numeri da nomi tipo '3-2.json' → (3, 2)
    """
    name = os.path.splitext(filename)[0]  # '3-2'
    parts = name.split("-")

    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        # fallback: file con nome non conforme
        return (float("inf"), name)


def merge_json_folder(input_dir, output_file):
    merged_data = []

    json_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".json")
    ]

    # 🔹 ordinamento numerico corretto
    json_files.sort(key=sort_key)

    for filename in json_files:
        file_path = os.path.join(input_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Errore nel file {filename}: {e}")
                continue

        # id = nome file senza estensione
        data["id"] = os.path.splitext(filename)[0]
        merged_data.append(data)

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(merged_data, out, ensure_ascii=False, indent=2)

    print(f"Creato file unico con {len(merged_data)} CBT ordinati → {output_file}")


if __name__ == "__main__":
    input_dir = "outputs_pipeline_unralated_transcription_sit2/generated_cbts/"
    output_file = "outputs_pipeline_unralated_transcription_sit2/merged_cbts.json"

    merge_json_folder(input_dir, output_file)
