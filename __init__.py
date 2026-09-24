# ComfyUI imports this directory as a package (spec_from_file_location on
# __init__.py), so __package__ is always set there. Imported on its own - as
# pytest does when it collects the repository root - there is no parent
# package for the relative import and nothing to register.
if __package__:
    from .ddrk_omega.sampler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
