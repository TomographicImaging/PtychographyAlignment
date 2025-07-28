def might_OpenViewer(request, data):
    if request.config.getoption("--view"):
        from viewer.OpenViewer import OpenViewer
        OpenViewer(data)