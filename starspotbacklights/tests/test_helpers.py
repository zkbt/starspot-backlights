from starspotbacklights import *

def test_basic_functions():
    deltaf = guesstimate_deltaf(f=0.2, f1=0.001)
    epsilon = calculate_eps(ratio=0.5, f=0.2)
    A = calculate_A(ratio=0.5, f=0.20, deltaf=0.01)
    f = solve_for_f(ratio=0.5, A=0.01, f1=0.001, visualize=True)

if __name__ == "__main__":
    outputs = {
        k.split("_")[-1]: v()
        for k, v in locals().items()
        if "test_" in k
    }
