from hstest import *


ANSWER = [0.554, 0.612, 0.881, 0.973]
# in ANSWER there are lower and upper bounds
# the correct answer is 0.583 0.927

class Eval2Test(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input the path to the files")
        output = pr.execute("test/data_stage9_train.csv test/data_stage9_test.csv").strip()
        try:
            res = [round(float(x), 3) for x in output.split()]
        except Exception:
            raise WrongAnswer("You should print two float values split with space.")
        if len(res) != 2:
            raise WrongAnswer("Wrong number of values. Print two numbers: true positives and true negatives normalized over the true rows.")
        if not (ANSWER[0] <= res[0] <= ANSWER[1]):
            raise WrongAnswer("Wrong true positives value (the first value).")
        if not (ANSWER[2] <= res[1] <= ANSWER[3]):
            raise WrongAnswer("Wrong true negatives value (the second value).")
        return CheckResult.correct()


if __name__ == '__main__':
    Eval2Test().run_tests()
