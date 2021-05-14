from starspotbacklights import *

def test_backlight():
    b = Backlight()
    b.set_parameters([3000, 2800, 0.25, 0.025, 0.005, 0.005])
    b.w_unspot, b.f_unspot

def test_filters():
    m = MEarth()
    m.center


if __name__ == "__main__":
    outputs = {
        k.split("_")[-1]: v()
        for k, v in locals().items()
        if "test_" in k
    }
