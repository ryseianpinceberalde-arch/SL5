"""Create MP_Data folders and save action labels."""

from sign_language_common import DATA_PATH, sanitize_action_name, update_actions_from_folders


def main():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Data folder ready: {DATA_PATH}")
    print("Type sign names separated by commas, or press ENTER to only refresh actions.json.")
    raw_names = input("Signs to create: ").strip()

    if raw_names:
        for raw_name in raw_names.split(","):
            action = sanitize_action_name(raw_name)
            if action:
                (DATA_PATH / action).mkdir(parents=True, exist_ok=True)
                print(f"Created/checked folder: MP_Data/{action}")

    actions = update_actions_from_folders()
    print(f"Saved labels to MP_Data/actions.json: {actions}")


if __name__ == "__main__":
    main()
