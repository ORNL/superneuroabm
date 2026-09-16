"""Step functions used only by the tests.

`test_registration_api` registers a user-supplied learning rule, so it needs a
rule that does not already live in `superneuroabm.step_functions`. This one was
previously imported from `examples/`; it lives here so the test does not depend
on example code that may be removed.
"""
