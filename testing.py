from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Assuming you have a pre-trained model saved in an H5 file
model = load_model('./model_v4.h5')

# Define the data generator
datagen = ImageDataGenerator(rescale=1./255)

# Provide the path to your test data
test_dir = './Dataset/test_good'

# Create a test data generator using flow_from_directory
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),  # Specify the target size of your images
    batch_size=64,
    class_mode='categorical'  # Choose the appropriate class mode based on your problem
)

# Evaluate the model on the test data
evaluation = model.evaluate(test_generator)

# Print the evaluation results
print("Test Loss: {:.2f}".format(evaluation[0]))
print("Test Accuracy: {:.2f}%".format(evaluation[1] * 100))
