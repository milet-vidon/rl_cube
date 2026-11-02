"""Encode captured RGB frames using the installed Isaac video encoder."""
import numpy as np

def encode_mp4(path,frames,fps=10):
    import omni.kit.app
    manager=omni.kit.app.get_app().get_extension_manager()
    if not manager.set_extension_enabled_immediate('omni.videoencoding',True):
        raise RuntimeError('Cannot enable installed video encoder')
    from video_encoding import get_video_encoding_interface
    api=get_video_encoding_interface()
    if api is None or not api.start_encoding(str(path),float(fps),len(frames),False):
        raise RuntimeError('Cannot start MP4 encoder')
    try:
        for frame in frames:
            rgba=np.asarray(frame.convert('RGBA'),dtype=np.uint8).copy()
            if not api.encode_next_frame_from_buffer(rgba):raise RuntimeError('MP4 frame encoding failed')
    finally:api.finalize_encoding()
