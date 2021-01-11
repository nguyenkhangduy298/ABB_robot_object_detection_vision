class Yolo:

    yoloConfigUrl = ""
    yoloWeight = ""
    yoloClass = ""

    def __init__(self, configUrl, weightUrl, classUrl):
        self.yoloConfigUrl = configUrl
        self.yoloWeight = weightUrl
        self.yoloClass = classUrl
