"""Run a non-destructive MP_Data health check."""

from sign_ai.training.dataset_checker import check_dataset, print_report


def main():
    print_report(check_dataset())


if __name__ == "__main__":
    main()

