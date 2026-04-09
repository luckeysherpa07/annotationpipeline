from annotation_feature import pipeline

def main():
    options = {
        "1": ("Run Pipeline", pipeline.run),
    }

    while True:
        print("\nChoose an option:")
        for key, (description, _) in options.items():
            print(f"{key}. {description}")
        print("0. Exit")

        choice = input("Enter the number: ").strip()
        if choice == "0":
            print("Exiting...")
            break
        elif choice in options:
            print(f"\nRunning {options[choice][0]}...\n")
            options[choice][1]()  # Call the function
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()