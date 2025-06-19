import boto3

collection_id = 'my-face-collection'  # Change as needed

import boto3

def list_all_faces(collection_id, region_name):
    rekognition = boto3.client('rekognition', region_name=region_name)
    faces = []
    next_token = None

    print(f"\n📥 Fetching faces from collection: {collection_id} (Region: {region_name})")

    while True:
        if next_token:
            response = rekognition.list_faces(
                CollectionId=collection_id,
                MaxResults=20,
                NextToken=next_token
            )
        else:
            response = rekognition.list_faces(
                CollectionId=collection_id,
                MaxResults=20
            )

        faces.extend(response['Faces'])

        for face in response['Faces']:
            print("-----------")
            print(f"FaceId: {face['FaceId']}")
            print(f"ImageId: {face['ImageId']}")
            print(f"ExternalImageId: {face.get('ExternalImageId', 'N/A')}")
            print(f"Confidence: {face['Confidence']:.2f}")
            print(f"BoundingBox: {face['BoundingBox']}")

        next_token = response.get('NextToken')
        if not next_token:
            break

    print(f"\n✅ Total faces found: {len(faces)}")


def delete_all_faces(collection_id, region_name):
    rekognition = boto3.client('rekognition', region_name=region_name)
    face_ids = []
    next_token = None

    print(f"\n🗑️ Fetching face IDs to delete from: {collection_id} (Region: {region_name})")

    # Collect all FaceIds
    while True:
        if next_token:
            response = rekognition.list_faces(
                CollectionId=collection_id,
                MaxResults=20,
                NextToken=next_token
            )
        else:
            response = rekognition.list_faces(
                CollectionId=collection_id,
                MaxResults=20
            )

        face_ids.extend([face['FaceId'] for face in response['Faces']])
        next_token = response.get('NextToken')
        if not next_token:
            break

    if not face_ids:
        print("ℹ️ No faces to delete.")
        return

    # Batch delete
    print(f"\n⚠️ Deleting {len(face_ids)} faces...")
    chunk_size = 1000
    for i in range(0, len(face_ids), chunk_size):
        batch = face_ids[i:i+chunk_size]
        rekognition.delete_faces(CollectionId=collection_id, FaceIds=batch)
        print(f"✅ Deleted {len(batch)} faces.")

    print("\n✅ All faces deleted successfully.")


if __name__ == "__main__":
    print("==== AWS Rekognition Face Manager ====\n")
    region = "ap-south-1"
    collection_id = "my-face-collection"

    print("\nChoose an option:")
    print("1. List all registered users")
    print("2. Delete all registered users (⚠️ irreversible)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        list_all_faces(collection_id, region)
    elif choice == '2':
        confirm = input("Are you sure you want to delete ALL faces? (yes/no): ").lower()
        if confirm == "yes":
            delete_all_faces(collection_id, region)
        else:
            print("❌ Deletion cancelled.")
    else:
        print("❌ Invalid choice.")
