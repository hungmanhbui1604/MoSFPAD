from torchvision import transforms

mosfpad_transforms = {
    "Train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, shear=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
    "Test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    ),
}

def get_transforms(transform_name: str):
    if transform_name == 'mosfpad':
        return mosfpad_transforms['Train'], mosfpad_transforms['Test']
    
    raise ValueError(f"Unknown tranform name: {transform_name}")
