
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def tokinized_data():
    
    def image_preprocessing():
        def image_puller(folder_path):
            img_data_array = []
            folder = folder_path
            
            for filename in os.listdir(folder):
                # Check if the file is a JPG image
                if filename.endswith(('.jpg', '.jpeg', '.png')):  # Added .png support
                    # Construct the full file path
                    img_path = os.path.join(folder, filename)
                    
                    # Read the image using OpenCV
                    img = cv2.imread(img_path)
                    
                    # Check if image loaded successfully
                    if img is not None:
                        # Resize for consistency (important!)
                        img = cv2.resize(img, (150, 150))
                        # Normalize pixel values to [0, 1]
                        img = img.astype(np.float32) / 255.0
                        img_data_array.append(img)
                    else:
                        print(f"Failed to load: {filename}")
            return img_data_array
        
        # TRAINING DATA - True cases (tumors)
        glioma_training = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Training\glioma_tumor')
        meningioma_training = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Training\meningioma_tumor')
        pituitary_training = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Training\pituitary_tumor')
        
        # Combine all tumor types for training (all are "True" / positive cases)
        yes_images_training = glioma_training + meningioma_training + pituitary_training
        
        # TRAINING DATA - False cases (no tumor)
        no_images_training = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Training\no_tumor')
        
        # TESTING DATA - True cases (tumors)
        glioma_testing = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Testing\glioma_tumor')
        meningioma_testing = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Testing\meningioma_tumor')
        pituitary_testing = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Testing\pituitary_tumor')
        
        # Combine all tumor types for testing
        yes_images_testing = glioma_testing + meningioma_testing + pituitary_testing
        
        # TESTING DATA - False cases (no tumor)
        no_images_testing = image_puller(r'C:\Users\install\Desktop\ML\Img\archive(1)\Testing\no_tumor')
        
        # Print dataset statistics
        
        
        def img_batcher(image_array):
            img_batch = []
            for img in image_array:
                # Reshape the image to keep dimensions (1, height, width, channels)
                # This preserves spatial structure
                if len(img.shape) == 3:
                    img_batch.append(img.reshape(1, *img.shape))
                else:
                    img_batch.append(img.reshape(1, -1))
            return img_batch
        
        yes_images_training_batch = img_batcher(yes_images_training)
        no_images_training_batch = img_batcher(no_images_training)
        yes_images_testing_batch = img_batcher(yes_images_testing)
        no_images_testing_batch = img_batcher(no_images_testing)
        
        return (yes_images_training_batch, no_images_training_batch, 
                yes_images_testing_batch, no_images_testing_batch)
    
    def image_tokenizer(training_yes, training_no, testing_yes, testing_no):
        def flatten_images(image_batch):
            flat_images = []
            target_length = 80
            
            for img in image_batch:
                # Flatten the image to 1D
                flat_img = img.reshape(-1)
                n = len(flat_img)
                
                # Calculate chunk size
                chunk_size = n / target_length
                
                # Create indices for averaging windows
                indices = np.floor(np.arange(target_length + 1) * chunk_size).astype(int)
                
                # Average each segment
                downsampled = np.array([
                    np.mean(flat_img[indices[i]:indices[i+1]]) 
                    for i in range(target_length)
                ])
                
                flat_images.append(downsampled)
            
            return flat_images
        
        flat_yes_train = flatten_images(training_yes)
        flat_no_train = flatten_images(training_no)
        flat_yes_test = flatten_images(testing_yes)
        flat_no_test = flatten_images(testing_no)
        
        # Create labeled datasets
        training_data = create_labeled_dataset(flat_yes_train, flat_no_train)
        testing_data = create_labeled_dataset(flat_yes_test, flat_no_test)
        
        # Shuffle both datasets
        training_data = shuffle_dataset(training_data)
        testing_data = shuffle_dataset(testing_data)
        
        return training_data, testing_data
    
    def create_labeled_dataset(yes_data, no_data, label_yes=1, label_no=0):
        dataset = []
        for img in yes_data:
            dataset.append([img, label_yes])
        for img in no_data:
            dataset.append([img, label_no])
        return dataset
    
    def shuffle_dataset(dataset):
        shuffled = dataset.copy()
        random.shuffle(shuffled)
        return shuffled
    
    training_data, testing_data = image_tokenizer(*image_preprocessing())
    return training_data, testing_data




def MLP_(training_data, testing_data,epochs=50000, learning_rate=0.01):


    training_data, testing_data = tokinized_data()
    X_train = np.array([data[0] for data in training_data])
    Y_train = np.array([data[1] for data in training_data]).reshape(-1, 1)
    X_test = np.array([data[0] for data in testing_data])
    Y_test = np.array([data[1] for data in testing_data]).reshape(-1, 1)
    def forward_pass(X, W1, b1, W2, b2, Y):

        def Relu(x):


            return np.maximum(0, x)
        def Sigmoid(x):
            return 1 / (1 + np.exp(-x))
        z1 = np.dot(X, W1) + b1
        hidden = Relu(z1)

        z2 = np.dot(hidden, W2) + b2
        output = Sigmoid(z2)
        epsilon = 1e-7
        loss = -np.mean(Y * np.log(output + epsilon) + 
                (1 - Y) * np.log(1 - output + epsilon))
    
        return hidden, output, loss, z1, z2


    def backprop(hidden, output, X, Y, W1, W2,z1, z2):

        m = X.shape[0]
        d_loss_d_z2 = (output - Y) / m
        d_loss_d_W2 = np.dot(hidden.T, d_loss_d_z2)
        d_loss_d_b2 = np.sum(d_loss_d_z2, axis=0, keepdims=True)
        d_loss_d_hidden = np.dot(d_loss_d_z2, W2.T)
        d_loss_d_z1 = d_loss_d_hidden * (z1 > 0).astype(float)
        d_loss_d_W1 = np.dot(X.T, d_loss_d_z1)
        d_loss_d_b1 = np.sum(d_loss_d_z1, axis=0, keepdims=True)
        return d_loss_d_W1, d_loss_d_b1, d_loss_d_W2, d_loss_d_b2
    
    def train(X,Y):

        W1 = np.random.rand(X.shape[1], 64) * 0.01
        b1 = np.zeros((1, 64))
        W2 = np.random.rand(64, 1) * 0.01
        b2 = np.zeros((1, 1))

        loss_history = []
        for epoch in range(epochs):
            hidden, output, loss, z1, z2 = forward_pass(X, W1, b1, W2, b2, Y)
            d_loss_d_W1, d_loss_d_b1, d_loss_d_W2, d_loss_d_b2 = backprop(hidden, output, X, Y, W1, W2,z1, z2)
            W1 -= learning_rate * d_loss_d_W1
            b1 -= learning_rate * d_loss_d_b1
            W2 -= learning_rate * d_loss_d_W2
            b2 -= learning_rate * d_loss_d_b2
           
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
                loss_history.append(loss)
           
        plt.plot(loss_history)
        plt.xlabel('Epochs (x10)')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')            
        plt.show()  
        return W1, b1, W2, b2  
    W1, b1, W2, b2 = train(X_train, Y_train)
    def test_model(X_test, Y_test, W1, b1, W2, b2):

   
    # Forward pass (same as training, but no gradients needed)
        def forward_pass_test(X, W1, b1, W2, b2):
            def Relu(x):
                return np.maximum(0, x)
            def Sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            z1 = np.dot(X, W1) + b1
            hidden = Relu(z1)
            z2 = np.dot(hidden, W2) + b2
            output = Sigmoid(z2)
            return output
        
        # Get predictions
        predictions = forward_pass_test(X_test, W1, b1, W2, b2)
        
        # Convert to binary classes
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == Y_test)
        
        # Loss (optional, for comparison)
        epsilon = 1e-7
        loss = -np.mean(Y_test * np.log(predictions + epsilon) + 
                        (1 - Y_test) * np.log(1 - predictions + epsilon))
        
        return {
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'accuracy': accuracy,
            'loss': loss
        }
    test_results = test_model(X_test, Y_test, W1, b1, W2, b2)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
MLP_(tokinized_data()[0], tokinized_data()[1])

