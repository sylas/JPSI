from perceptron import simple_perceptron
import argparse
import numpy as np

if __name__ == '__main__':

    # Parse the arguments.
    # They can be also read from a file (@file.par)
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@')
    ap.add_argument('-t', '--testset', help='Path to test dataset',
        metavar='filename', required=True)
    ap.add_argument('-m', '--model', help='Path to trained model',
        metavar='filename', required=True)

    args = vars(ap.parse_args())

    test_dataset_filename = args['testset']
    model_filename = args['model']

    # Create instance of the simple_perceptron class
    # (without any parameters, since only prediction is to be done)
    p = simple_perceptron()

    # Load previously saved model
    p.load_model(model_filename)

    # Read test data and test the perceptron
    # with the trained weights
    Xtest,Yexpected = p.read_input_data(test_dataset_filename)
    Yout = p.test(Xtest)

    Xtest, Yout, Yexpected = p.unnormalize(Xtest, Yout, Yexpected)

    print()
    print('Test results:')

    for i in range(len(Yout)):
        # For summation only:
        print('{:.3f} + {:.3f} = {:.3f} (expected {:.3f})'
                .format(Xtest[i][0],
                        Xtest[i][1],
                        Yout[i],
                        Yexpected[i]))
        # General:
        #print('obtained: {}, expected: {}'
        #    .format(Yout[i], Yexpected[i]))


    # Scores: RMSE and R squared score
    sse = sum((np.array(Yexpected) - np.array(Yout))**2)
    tse = (len(Yexpected) - 1) * np.var(Yexpected, ddof=1)
    rmse = np.sqrt(sse / len(Yout))
    r2_score = 1 - (sse / tse)
    print("\nRMSE score      = {:.2f}".format(rmse))
    print("R squared score = {:.2f}".format(r2_score))



