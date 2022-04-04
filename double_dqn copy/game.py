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
    def __init__(self, game_url, chrome_driver_path, init_script):
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
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
        self.driver.execute_script("Runner.config.ACCELERATION=0") # no ACCELERATION and birds for the game, easy mode, 
        time.sleep(1) # wait the html
        self.driver.execute_script(init_script) # set id for the canvas
        self.CV_display = self.show_img()  # show the state using opencv instead of the browser
        self.CV_display.__next__() # initiliaze the display coroutine
    
    def screen_shot(self):
        image_b64 = self.driver.execute_script(self.getbase64Script)
        np_img = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY) # change 4 channels to 1 gray channel
        np_img = cv2.resize(np_img, (80,80)) # resize the image to smaller
        #np_img = Image.fromarray(np_img)
        #np_img = np_img.save('./img/'+str(i)+'.png')
        return np_img

    def show_img(self, graphs = False):
        while True:
            screen = (yield)
            window_title = "logs" if graphs else "game_play"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
            imS = cv2.resize(screen, (200, 130)) # the size of the cv2 window
            cv2.imshow(window_title, imS)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break
            
    # get current state
    def get_state(self,actions):
        reward = 0.1
        is_over = False #game over
        if actions[1] == 1:
            self.press_up()
        '''elif actions[1] == 2:
            self.press_down()'''
        image = self.screen_shot()
        self.CV_display.send(image)
        if self.get_crashed():
            reward = -1
            is_over = True
        
        return image, reward, is_over #return the Experience tuple
    #  Runner.instance_.horizon.obstacles
    #  Runner.instance_.runningTime > Runner.instance_.config.CLEAR_TIME # true if obstacle

    # check pass the obstacle
    # console.log(Runner.instance_.horizon.obstacles[0].xPos+1 + (Runner.instance_.horizon.obstacles[0].typeConfig.width*Runner.instance_.horizon.obstacles[0].size-2), Runner.instance_.tRex.xPos+1+Runner.instance_.tRex.config.WIDTH-2)

    '''def get_score(self):
        return self.driver.execute_script("Runner.instance_.distanceMeter.getActualDistance\
                                          (Math.ceil(Runner.instance_.distanceRan));")'''

    def get_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return self.driver.execute_script("return Runner.instance_.playing")
    def restart(self):
        self.driver.execute_script("Runner.instance_.restart()")
    def press_up(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    def get_score(self):
        score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)
    def pause(self):
        return self.driver.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self.driver.execute_script("return Runner.instance_.play()")
    def end(self):
        self.driver.close()