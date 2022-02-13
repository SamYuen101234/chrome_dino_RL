from selenium import webdriver
from selenium.webdriver.chrome.options import Options
#from selenium.webdriver.common.keys import Keys
import time
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2
from selenium.webdriver.common.keys import Keys
class Game():
    def __init__(self, ):
        game_url = "http://www.aboutsamyuen.com/projects/chrome_dino_js/index.html" # 
        chrome_driver_path = "/usr/local/bin/chromedriver"
        self.init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self.getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"  
        chrome_options = Options()
        chrome_options.add_argument('headless') # not showing browser is faster
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self.driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
        self.driver.set_window_position(x=-10,y=0)
        # print(self.driver.get_window_size()) # print the size of browser
        self.driver.set_window_size(1450, 1080)
        self.driver.get(game_url)
        self.driver.execute_script("Runner.config.ACCELERATION=0") # no ACCELERATION for the game
        time.sleep(1) # wait the html
        self.driver.execute_script(self.init_script) # set id for the canvas
        #self.CV_display = self.show_img()  # show the state using opencv instead of the browser
        #self.CV_display.__next__() # initiliaze the display coroutine
        #self.writer = tb_writer
        self.screen_shot()
    
    def screen_shot(self):
        #self.driver.execute_script(self.init_script)
        image_b64 = self.driver.execute_script(self.getbase64Script)
        np_img = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY) # change 4 channels to 1 gray channel
        np_img = cv2.resize(np_img, (80,80)) # resize the image to smaller
        np_img = Image.fromarray(np_img)
        np_img = np_img.save('./img/temp.png')
        return np_img

game = Game()
for i in range(100):
    print(i)
    game.screen_shot()