"""A simple GUI to view and replay the collected demos."""


from tools.demo_player.demo_player_window import DemoPlayerWindow
from tools.shared.primary_window import PrimaryWindow


class DemoPlayer:
    """A simple GUI to view and replay the collected demos."""

    def __init__(self):
        """Init."""
        PrimaryWindow(DemoPlayerWindow, title="Demo Player", resizable=True)


if __name__ == "__main__":
    demo_player = DemoPlayer()
