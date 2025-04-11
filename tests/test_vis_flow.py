import numpy as np
import pytest
import matplotlib.pyplot as plt
from openopticalflow.vis_flow import vis_flow

def test_vis_flow_basic():
    """Test that vis_flow creates a figure with expected properties."""
    # Create a simple velocity field
    vx = np.ones((10, 10))
    vy = np.ones((10, 10))

    # Call vis_flow
    ax = vis_flow(vx, vy)

    # Check that an axis was returned
    assert isinstance(ax, plt.Axes)

    # Check that the axis contains a quiver plot
    quiver_artists = [artist for artist in ax.get_children()
                     if isinstance(artist, plt.matplotlib.quiver.Quiver)]
    assert len(quiver_artists) == 1

    # Clean up
    plt.close(plt.gcf())

def test_vis_flow_with_nan():
    """Test that vis_flow handles NaN values correctly."""
    # Create a velocity field with NaN values
    vx = np.ones((10, 10))
    vy = np.ones((10, 10))
    vx[5, 5] = np.nan
    vy[5, 5] = np.nan

    # Call vis_flow
    ax = vis_flow(vx, vy)

    # Check that an axis was returned
    assert isinstance(ax, plt.Axes)

    # Clean up
    plt.close(plt.gcf())

def test_vis_flow_parameters():
    """Test that vis_flow parameters affect the output."""
    # Create a simple velocity field
    vx = np.ones((20, 20))
    vy = np.ones((20, 20))

    # Call vis_flow with different colors
    ax1 = vis_flow(vx, vy, color='red')
    fig1 = plt.gcf()

    plt.figure()  # Create a new figure
    ax2 = vis_flow(vx, vy, color='blue')
    fig2 = plt.gcf()

    # Get the quiver artists
    quiver1 = [artist for artist in ax1.get_children()
              if isinstance(artist, plt.matplotlib.quiver.Quiver)][0]
    quiver2 = [artist for artist in ax2.get_children()
              if isinstance(artist, plt.matplotlib.quiver.Quiver)][0]

    # The quivers should have different colors
    assert quiver1.get_facecolor()[0, 0] != quiver2.get_facecolor()[0, 0]

    # Clean up
    plt.close(fig1)
    plt.close(fig2)

def test_vis_flow_zero_field():
    """Test that vis_flow handles zero velocity field."""
    # Create a zero velocity field
    vx = np.zeros((10, 10))
    vy = np.zeros((10, 10))

    # Call vis_flow
    ax = vis_flow(vx, vy)

    # Check that an axis was returned
    assert isinstance(ax, plt.Axes)

    # Clean up
    plt.close(plt.gcf())
