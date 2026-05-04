# Block ROS2 pytest plugins that conflict with standard pytest
collect_ignore_glob = []


def pytest_configure(config):
    """Disable ROS2/launch_testing plugins to avoid conflicts."""
    for plugin_name in [
        "launch_testing_ros_pytest_entrypoint",
        "launch_testing",
        "ament_copyright",
        "ament_pep257",
        "ament_flake8",
        "ament_xmllint",
        "ament_lint",
    ]:
        try:
            config.pluginmanager.set_blocked(plugin_name)
        except Exception:
            pass
