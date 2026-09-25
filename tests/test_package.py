from importlib.metadata import entry_points


def test_mjlab_finds_the_package():
    found = {ep.name: ep.value for ep in entry_points(group="mjlab.tasks")}
    assert found.get("microduck_challenges") == "microduck_challenges"


def test_the_arena_is_installed_from_the_workshop_group():
    from arena.manifest import load_event

    assert load_event("sprint-2m").id == "sprint-2m"
