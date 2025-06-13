import matplotlib.pyplot as plt


def updatePercentageComplete(percentageComplete, barHandle, handles):
    """
    Update the progress bar in the graphical user interface.

    Args:
        percentageComplete (float): Current percentage of completion (0 to 1).
        barHandle: Handle to the progress bar (matplotlib object).
        handles: Dictionary of handles to GUI elements.
    """
    # Update the YData of the bar to reflect the new percentage
    barHandle.set_height(percentageComplete)

    # Optionally update the axis (if required for custom styling)
    # plt.axis([0, 1, 0, 0.1])  # Commented out equivalent MATLAB lines
    # plt.axis('off')  # Hide axis if desired

    plt.draw()  # Refresh the plot
    plt.pause(0.001)  # Pause briefly to allow for UI update
