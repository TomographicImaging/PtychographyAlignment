def might_OpenViewer(request, data):
    if request.config.getoption("--viewer"):
        from viewer.OpenViewer import OpenViewer
        OpenViewer(data)