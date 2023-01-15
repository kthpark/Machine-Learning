from hstest import *
from math import isclose


def is_float(num: str):
    try:
        float(num)
        return True
    except ValueError:
        return False


class Stage5Test(StageTest):
    def check_outputs_number(self, values_number: int, user_output: str):
        outputs = user_output.split()

        if not all(is_float(output) for output in outputs):
            raise WrongAnswer(f"Answer '{user_output}' contains non-numeric values.")

        if len(outputs) != values_number:
            raise WrongAnswer(f"A wrong number of values, read '{len(outputs)}', expected '{values_number}'.")

    def check_num_values(self, values: list, user_values: list, message: str, rel_tol=1.0e-2):
        if not all(isclose(value, user_value, rel_tol=rel_tol) for value, user_value in zip(values, user_values)):
            raise WrongAnswer(message)

    @dynamic_test(time_limit=600000)
    def test(self):
        pr = TestedProgram()
        user_output = pr.start().strip()

        if len(user_output.strip()) == 0:
            raise WrongAnswer("Seems like your program does not show any output.")

        # check output format
        self.check_outputs_number(2, user_output)

        # check accuracy after PCA
        answer = [0.85632]
        user_values = [float(value) for value in user_output.split()][:1]
        self.check_num_values(answer, user_values,
                              "The accuracy on the test set after PCA optimization is wrong.\n"
                              "Check the parameters of your estimator LogisticRegression(solver='liblinear').\n"
                              "Make sure that you provide numbers in the correct order.")

        # check AUC after PCA
        answer = [0.93701]
        user_values = [float(value) for value in user_output.split()][1:2]
        self.check_num_values(answer, user_values,
                              "The AUC on the test set after PCA optimization is wrong.\n"
                              "Check the parameters of your estimator LogisticRegression(solver='liblinear').\n"
                              "Make sure that you provide numbers in the correct order.")

        return CheckResult.correct()


if __name__ == '__main__':
    Stage5Test().run_tests()
