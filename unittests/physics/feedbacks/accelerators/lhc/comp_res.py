import numpy as np

from blond.handle_results.helpers import callers_relative_path


def main():
    birk_b2 = np.load(
        callers_relative_path(
            "resources/lhc_convergence_to_steadystate_40.0deg.npz",
            stacklevel=2,
        )
    )
    egon_b2 = np.load(
        callers_relative_path(
            "lhc_convergence_to_steadystate_40.0deg.npz", stacklevel=2
        )
    )

    # for el_to_check in ["rf_beam_current", "rf_beam_current_phase"]:
    #     print(f"checking {el_to_check}")
    #     try:
    #         assert np.testing.assert_allclose(np.abs(birk_b2[el_to_check]), egon_b2[el_to_check], rtol=1e-5, )
    #     except AssertionError as e:
    #         print(e)
    try:
        np.testing.assert_allclose(
            birk_b2["rf_beam_current_phase"] + 10,
            egon_b2["rf_beam_current_phase"] + 10,
            rtol=4e-5,
            err_msg="Error in turn-by-turn phase of rf beam current",
        )
    except AssertionError as e:
        print(e)

    try:
        np.testing.assert_allclose(
            np.abs(birk_b2["rf_beam_current"]),
            np.abs(egon_b2["rf_beam_current"]),
            rtol=1e-5,
            err_msg="Error in absolute value of rf beam current",
        )
    except AssertionError as e:
        print(e)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(np.angle(birk_b2["rf_beam_current"][341], deg=True), ls="--")
    plt.plot(np.angle(egon_b2["rf_beam_current"][341], deg=True))
    plt.show(block=False)

    plt.clf()
    plt.plot(np.abs(birk_b2["rf_beam_current"][341]), ls="--")
    plt.plot(np.abs(egon_b2["rf_beam_current"][341]))
    plt.show(block=False)

    try:
        np.testing.assert_allclose(
            np.angle(birk_b2["rf_beam_current"], deg=True),
            np.angle(egon_b2["rf_beam_current"], deg=True),
            rtol=2e-3,
            err_msg="Error in phase of rf beam current",
        )
    except AssertionError as e:
        print(e)

    try:
        np.testing.assert_allclose(
            np.abs(birk_b2["rf_voltage"]),
            np.abs(egon_b2["rf_voltage"]),
            rtol=9e-6,
            err_msg="Error in absolute value of rf voltage",
        )
    except AssertionError as e:
        print(e)

    try:
        np.testing.assert_allclose(
            np.angle(birk_b2["rf_voltage"], deg=True) + 10,
            np.angle(egon_b2["rf_voltage"], deg=True) + 10,
            rtol=4e-5,
            err_msg="Error in phase value of rf voltage",
        )
    except AssertionError as e:
        print(e)

    try:
        np.testing.assert_allclose(
            np.abs(birk_b2["rf_power"]),
            np.abs(egon_b2["rf_power"]),
            rtol=2e-3,
            err_msg="Error in absolute value of rf power",
        )
    except AssertionError as e:
        print(e)

    try:
        np.testing.assert_allclose(
            np.angle(birk_b2["rf_power"], deg=True),
            np.angle(egon_b2["rf_power"], deg=True),
            atol=1e-9,
            err_msg="Error in absolute value of rf power",
        )
    except AssertionError as e:
        print(e)
    plt.show()


if __name__ == "__main__":
    main()
