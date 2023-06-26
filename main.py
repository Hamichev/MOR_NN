import pygame
from math import sqrt
import numpy as np
import pickle
from os import listdir
import random
import keras

#from tensorflow import keras

model = keras.models.load_model('C:\\Users\\Hamichev\\Desktop\\Learn\\MOR\\project\\Move_Robot_NN\\model')

def r_math(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

data_NN = list()
ans_NN = list()


r1_value_flag = 1
r2_value_flag = 1
r3_value_flag = 1
lu_value_flag = 1
ld_value_flag = 1
ll_value_flag = 1
lr_value_flag = 1

l_data_NN = {}

def work_model():
    global robot_rect

    x_mouse, y_mouse = pygame.mouse.get_pos()
    x_mouse -= 2 * tag_param
    y_mouse -= tag_param
    create_data_for_NN(int(r_math(x0, y0, robot_rect[0] + 25, robot_rect[1] + 30)) * r1_value_flag,
                       int(r_math(x1, y1, robot_rect[0] + 25, robot_rect[1] + 30)) * r2_value_flag,
                       int(r_math(x2, y2, robot_rect[0] + 25, robot_rect[1] + 30)) * r3_value_flag,
                       int(robot_rect[1] - tag_param) * lu_value_flag,
                       int(H - robot_rect[1] - 4 * tag_param) * ld_value_flag,
                       int(robot_rect[0] - 2 * tag_param) * ll_value_flag,
                       int(W - robot_rect[0] - 4 * tag_param) * lr_value_flag,
                       int(x_mouse),
                       int(y_mouse),
                       [0])
    buf = np.array([data_NN[-1]])
    #print(buf)
    res = np.argmax(model.predict(buf))
    match (res):
        case 1:
            robot_rect.y -= speed
        case 2:
            robot_rect.x -= speed
        case 3:
            robot_rect.x -= speed
            robot_rect.y -= speed
        case 4:
            robot_rect.y += speed
        case 5:
            robot_rect.x -= speed
            robot_rect.y += speed
        case 6:
            robot_rect.x += speed
        case 7:
            robot_rect.x += speed
            robot_rect.y -= speed
        case 8:
            robot_rect.x += speed
            robot_rect.y += speed

def info_ans_NN():
    temp_dict = dict()
    for i in range(len(ans_NN)):
        if (str(ans_NN[i][0]) in temp_dict):
            temp_dict[str(ans_NN[i][0])] += 1
        else:
            temp_dict[str(ans_NN[i][0])] = 1

    for key, val in temp_dict.items():
        print(key, " : ", val)

def app_stop_list_in_data_NN():
    create_data_for_NN(int(r_math(x0, y0, robot_rect[0] + 25, robot_rect[1] + 30)) * r1_value_flag,
                       int(r_math(x1, y1, robot_rect[0] + 25, robot_rect[1] + 30)) * r2_value_flag,
                       int(r_math(x2, y2, robot_rect[0] + 25, robot_rect[1] + 30)) * r3_value_flag,
                       int(robot_rect[1] - tag_param) * lu_value_flag,
                       int(H - robot_rect[1] - 4 * tag_param) * ld_value_flag,
                       int(robot_rect[0] - 2 * tag_param) * ll_value_flag,
                       int(W - robot_rect[0] - 4 * tag_param) * lr_value_flag,
                       int(x_robot - tag_param*2),
                       int(y_robot - tag_param),
                       [0])#[0, 0, 0, 0])
    update_info_list_NN()

def rand_val_NN():

    global r1_value_flag
    global r2_value_flag
    global r3_value_flag
    global lu_value_flag
    global ld_value_flag
    global ll_value_flag
    global lr_value_flag

    global x_robot
    global y_robot

    global robot_rect

    for _ in range(1):
        robot_rect.x = random.randint(32, 766)
        robot_rect.y = random.randint(35, 498)
        for i in range(128 * 1):
            out = [1 if i & (1 << (7 - n)) else 0 for n in range(8)]
            r1_value_flag = out[7]
            r2_value_flag = out[6]
            r3_value_flag = out[5]
            lu_value_flag = out[4]
            ld_value_flag = out[3]
            ll_value_flag = out[2]
            lr_value_flag = out[1]
            x_mouse = random.randint(32, 766)
            y_mouse = random.randint(35, 498)
            x_robot, y_robot = getCoordsWithMatrix()
            create_data_for_NN(int(r_math(x0, y0, robot_rect[0] + 25, robot_rect[1] + 30)) * r1_value_flag,
                               int(r_math(x1, y1, robot_rect[0] + 25, robot_rect[1] + 30)) * r2_value_flag,
                               int(r_math(x2, y2, robot_rect[0] + 25, robot_rect[1] + 30)) * r3_value_flag,
                               int(robot_rect[1] - tag_param) * lu_value_flag,
                               int(H - robot_rect[1] - 4 * tag_param) * ld_value_flag,
                               int(robot_rect[0] - 2 * tag_param) * ll_value_flag,
                               int(W - robot_rect[0] - 4 * tag_param) * lr_value_flag,
                               int(x_mouse),
                               int(y_mouse),
                               calc_ans(x_robot - tag_param * 2, y_robot - tag_param, x_mouse, y_mouse))
            update_info_list_NN()
        for _ in range(32): #32 - для равнозначного количества данных
            robot_rect.x = random.randint(32, 766)
            robot_rect.y = random.randint(35, 498)
            x_robot, y_robot = getCoordsWithMatrix()
            app_stop_list_in_data_NN()

def update_info_list_NN():
    str_for_dict = ""
    if (ans_NN[-1][0] == 0): #and ans_NN[-1][1] == 0 and ans_NN[-1][2] == 0 and ans_NN[-1][3] == 0):
        str_for_dict = "stop_"

    if(str_for_dict == ""):
        for i, val in enumerate([r1_value_flag, r2_value_flag, r3_value_flag,
                                 lu_value_flag, ld_value_flag, ll_value_flag, lr_value_flag]):
            if(not val):
                if(str_for_dict == ""):
                    str_for_dict = "null_"
                match i:
                    case 0:
                        str_for_dict += "r1_"
                    case 1:
                        str_for_dict += "r2_"
                    case 2:
                        str_for_dict += "r3_"
                    case 3:
                        str_for_dict += "lu_"
                    case 4:
                        str_for_dict += "ld_"
                    case 5:
                        str_for_dict += "ll_"
                    case 6:
                        str_for_dict += "lr_"

    if(str_for_dict == ""):
        str_for_dict = "normal"
    else:
        str_for_dict = str_for_dict[:-1]

    if(str_for_dict in l_data_NN):
        l_data_NN[str_for_dict] += 1
    else:
        l_data_NN[str_for_dict] = 1

def print_info_list_NN():
    for key, val in l_data_NN.items():
        print(key, " : ", val)
def save_data_NN():
    lisdir_project = listdir('.')
    ind_train_file = 1
    ind_ans_file = 1
    for name in lisdir_project:
        if(name == "train_" + str(ind_train_file)):
            ind_train_file+=1
        if (name == "ans_" + str(ind_ans_file)):
            ind_ans_file += 1
    with open("train_"+str(ind_train_file), "wb") as fp:  # Pickling data
        pickle.dump(data_NN, fp)
    with open("ans_"+str(ind_ans_file), "wb") as fp:  # Pickling ans
        pickle.dump(ans_NN, fp)
    print("Data and ans NN save to pickle")

def calc_ans(temp_x_robot, temp_y_robot, temp_x_mouse, temp_y_mouse):
    temp_ans = [0, 0, 0, 0]
    res = 0
    if(temp_x_robot < temp_x_mouse):
        temp_ans[0] = 1
    elif(temp_x_robot > temp_x_mouse):
        temp_ans[2] = 1
    if (temp_y_robot < temp_y_mouse):
        temp_ans[1] = 1
    elif (temp_y_robot > temp_y_mouse):
        temp_ans[3] = 1
    match(temp_ans):
        case[0, 0, 0, 1]:
            res = 1
        case [0, 0, 1, 0]:
            res = 2
        case [0, 0, 1, 1]:
            res = 3
        case [0, 1, 0, 0]:
            res = 4
        case [0, 1, 1, 0]:
            res = 5
        case [1, 0, 0, 0]:
            res = 6
        case [1, 0, 0, 1]:
            res = 7
        case [1, 1, 0, 0]:
            res = 8
    return [res]#temp_ans

def create_data_for_NN(r1, r2, r3, lu, ld, ll, lr, nc_x, nc_y, ans_data):
    data_NN.append([r1, r2, r3, lu, ld, ll, lr, nc_x, nc_y])
    ans_NN.append(ans_data)
    #print("data_NN: ", data_NN)
    #print("asn_NN: ", asn_NN)

def getCoordsWithMatrix():

    A = np.matrix([[-2 * x0, -2 * y0, 1],
                   [-2 * x1, -2 * y1, 1],
                   [-2 * x2, -2 * y2, 1]])

    B = np.linalg.inv(A)

    dist = np.array([r_math(x0, y0, robot_rect[0] + 25, robot_rect[1] + 30),
                     r_math(x1, y1, robot_rect[0] + 25, robot_rect[1] + 30),
                     r_math(x2, y2, robot_rect[0] + 25, robot_rect[1] + 30)])

    y = np.matrix([[(dist[0] ** 2) - (x0 ** 2) - (y0 ** 2)],
                   [(dist[1] ** 2) - (x1 ** 2) - (y1 ** 2)],
                   [(dist[2] ** 2) - (x2 ** 2) - (y2 ** 2)]])

    res = B.dot(y)
    result = np.squeeze(np.asarray(res))

    return result[0], result[1]

pygame.init()

#-------------------Values--------------------------
W, H = 900, 600 #-> само поле W - tag_param*4, H - tag_param*3 -> 800, 525
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
tag_param = 25
x0, y0 = tag_param, 55
x1, y1 = tag_param, H - tag_param*3
x2, y2 = W - tag_param, H // 2
speed = 10
check_q = 0
check_w = 0
text = pygame.font.SysFont(None, 30)
text_r = pygame.font.SysFont(None, 28)
text_r1 = text_r.render('r1', 1, GREEN, WHITE)
text_r2 = text_r.render('r2', 1, GREEN, WHITE)
text_r3 = text_r.render('r3', 1, GREEN, WHITE)
text_lu = text_r.render('lu', 1, GREEN, WHITE)
text_ld = text_r.render('ld', 1, GREEN, WHITE)
text_ll = text_r.render('ll', 1, GREEN, WHITE)
text_lr = text_r.render('lr', 1, GREEN, WHITE)
#---------------------------------------------------

sc = pygame.display.set_mode((W, H))
pygame.display.set_caption('Tags_system')
FPS = 60
clock = pygame.time.Clock()

#-----------------------------Image--------------------
pole_image = pygame.transform.scale(pygame.image.load('Pole.png'), (W - tag_param*4, H - tag_param*3))
#------------------------------------------------------


#--------------------------ROBOT--------------------
robot = pygame.Surface((52, 52))
robot_rect = robot.get_rect(center=(250, 250))
robot.fill(WHITE)
pygame.draw.aaline(robot, BLACK, (0, 50), (25, 0))
pygame.draw.aaline(robot, BLACK, (25, 0), (50, 50))
pygame.draw.aaline(robot, BLACK, (50, 50), (0, 50))
#---------------------------------------------------

flag_rand_active = -1
flag_model = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                if(r1_value_flag):
                    r1_value_flag = 0
                    text_r1 = text_r.render('r1', 1, RED, WHITE)
                else:
                    r1_value_flag = 1
                    text_r1 = text_r.render('r1', 1, GREEN, WHITE)
                print("r1 = ", r1_value_flag)
            if event.key == pygame.K_2:
                if (r2_value_flag):
                    r2_value_flag = 0
                    text_r2 = text_r.render('r2', 1, RED, WHITE)
                else:
                    r2_value_flag = 1
                    text_r2 = text_r.render('r2', 1, GREEN, WHITE)
                print("r2 = ", r2_value_flag)
            if event.key == pygame.K_3:
                if (r3_value_flag):
                    r3_value_flag = 0
                    text_r3 = text_r.render('r3', 1, RED, WHITE)
                else:
                    r3_value_flag = 1
                    text_r3 = text_r.render('r3', 1, GREEN, WHITE)
                print("r3 = ", r3_value_flag)
            if event.key == pygame.K_4:
                if (lu_value_flag):
                    lu_value_flag = 0
                    text_lu = text_r.render('lu', 1, RED, WHITE)
                else:
                    lu_value_flag = 1
                    text_lu = text_r.render('lu', 1, GREEN, WHITE)
                print("lu = ", lu_value_flag)
            if event.key == pygame.K_5:
                if (ld_value_flag):
                    ld_value_flag = 0
                    text_ld = text_r.render('ld', 1, RED, WHITE)
                else:
                    ld_value_flag = 1
                    text_ld = text_r.render('ld', 1, GREEN, WHITE)
                print("ld = ", ld_value_flag)
            if event.key == pygame.K_6:
                if (ll_value_flag):
                    ll_value_flag = 0
                    text_ll = text_r.render('ll', 1, RED, WHITE)
                else:
                    ll_value_flag = 1
                    text_ll = text_r.render('ll', 1, GREEN, WHITE)
                print("ll = ", ll_value_flag)
            if event.key == pygame.K_7:
                if (lr_value_flag):
                    lr_value_flag = 0
                    text_lr = text_r.render('lr', 1, RED, WHITE)
                else:
                    lr_value_flag = 1
                    text_lr = text_r.render('lr', 1, GREEN, WHITE)
                print("lr = ", lr_value_flag)
            if event.key == pygame.K_q:
                if check_q == 0:
                    check_q = 1
                else:
                    check_q = 0
            if event.key == pygame.K_w:
                if check_w == 0:
                    check_w = 1
                else:
                    check_w = 0
            if event.key == pygame.K_s:
                save_data_NN()
            if event.key == pygame.K_r:
                print("Start random data NN")
                flag_rand_active = 10000
            if event.key == pygame.K_p:
                app_stop_list_in_data_NN()
            if event.key == pygame.K_o:
                info_ans_NN()
            if event.key == pygame.K_m:
                if(not flag_model):
                    print("model_on")
                    flag_model = 1
                else:
                    print("model_off")
                    flag_model = 0
            if event.key == pygame.K_i:
                print_info_list_NN()
            if event.key == pygame.K_SPACE:
                x_mouse, y_mouse = pygame.mouse.get_pos()
                x_mouse -= 2*tag_param
                y_mouse -= tag_param
                create_data_for_NN(int(r_math(x0, y0, robot_rect[0] + 25, robot_rect[1] + 30))*r1_value_flag,
                                   int(r_math(x1, y1, robot_rect[0] + 25, robot_rect[1] + 30))*r2_value_flag,
                                   int(r_math(x2, y2, robot_rect[0] + 25, robot_rect[1] + 30))*r3_value_flag,
                                   int(robot_rect[1] - tag_param)*lu_value_flag,
                                   int(H - robot_rect[1] - 4 * tag_param)*ld_value_flag,
                                   int(robot_rect[0] - 2 * tag_param)*ll_value_flag,
                                   int(W - robot_rect[0] - 4 * tag_param)*lr_value_flag,
                                   int(x_mouse),
                                   int(y_mouse),
                                   calc_ans(x_robot - tag_param*2, y_robot - tag_param, x_mouse, y_mouse))
                update_info_list_NN()

    if (flag_model):
        work_model()
    if(flag_rand_active > 0):
        rand_val_NN()
        flag_rand_active-=1
    elif(flag_rand_active == 0):
        print("Random data NN done!")
        flag_rand_active = -1
    key_pressed = pygame.key.get_pressed()
    if key_pressed[pygame.K_LEFT]:
        robot_rect.x -= speed
    if key_pressed[pygame.K_RIGHT]:
        robot_rect.x += speed
    if key_pressed[pygame.K_UP]:
        robot_rect.y -= speed
    if key_pressed[pygame.K_DOWN]:
        robot_rect.y += speed

    if robot_rect.top <= tag_param + 5:
        robot_rect.top = tag_param + 5
    if robot_rect.bottom >= H - tag_param*2 - 5:
        robot_rect.bottom = H - tag_param*2 - 5
    if robot_rect.left <= tag_param*2 + 7:
        robot_rect.left = tag_param*2 + 7
    if robot_rect.right >= W - tag_param*2 - 7:
        robot_rect.right = W - tag_param*2 - 7

    sc.fill(WHITE)

    if check_w:
        sc.blit(pole_image, (tag_param*2, tag_param))
    else:
        pygame.draw.rect(sc, BLACK, (tag_param*2, tag_param, W - tag_param*4, H - tag_param*3), 2) #(surface, (red, green, blue), (x, y, width, height), 2)

    sc.blit(robot, robot_rect)

    if check_q:
        pygame.draw.circle(sc, BLACK, (x0, y0), int(r_math(x0, y0, robot_rect[0]+25, robot_rect[1]+30)), 1)
        pygame.draw.circle(sc, BLACK, (x1, y1), int(r_math(x1, y1, robot_rect[0]+25, robot_rect[1]+30)), 1)
        pygame.draw.circle(sc, BLACK, (x2, y2), int(r_math(x2, y2, robot_rect[0]+25, robot_rect[1]+30)), 1)

    pygame.draw.circle(sc, BLACK, (x0, y0), tag_param, 2)
    pygame.draw.circle(sc, BLACK, (x1, y1), tag_param, 2)
    pygame.draw.circle(sc, BLACK, (x2, y2), tag_param, 2)

    pygame.draw.aaline(sc, BLACK, (x0, y0), (robot_rect[0]+25, robot_rect[1]+30))
    pos_r1 = text_r1.get_rect(center=((robot_rect[0]+25 + x0) // 2,
                                      (robot_rect[1]+30 + y0) // 2))
    sc.blit(text_r1, pos_r1)

    pygame.draw.aaline(sc, BLACK, (x1, y1), (robot_rect[0]+25, robot_rect[1]+30))
    pos_r2 = text_r2.get_rect(center=((robot_rect[0] + 25 + x1) // 2,
                                      (robot_rect[1] + 30 + y1) // 2))
    sc.blit(text_r2, pos_r2)

    pygame.draw.aaline(sc, BLACK, (x2, y2), (robot_rect[0]+25, robot_rect[1]+30))
    pos_r3 = text_r3.get_rect(center=((robot_rect[0] + 25 + x2) // 2,
                                      (robot_rect[1] + 30 + y2) // 2))
    sc.blit(text_r3, pos_r3)

    #lu
    lu_y = tag_param
    pygame.draw.aaline(sc, BLACK, (robot_rect[0] + 25, tag_param), (robot_rect[0] + 25, robot_rect[1] + 30))
    pos_lu = text_lu.get_rect(center=(robot_rect[0] + 25,
                                     (robot_rect[1] + 30 + tag_param) // 2))
    sc.blit(text_lu, pos_lu)

    # ld
    ld_y = H - tag_param*2 - 2
    pygame.draw.aaline(sc, BLACK, (robot_rect[0] + 25, H - tag_param*2 - 2), (robot_rect[0] + 25, robot_rect[1] + 30))
    pos_ld = text_ld.get_rect(center=(robot_rect[0] + 25,
                                      (robot_rect[1] + 30 + H - tag_param*2 - 2) // 2))
    sc.blit(text_ld, pos_ld)

    # ll
    ll_x = tag_param*2
    pygame.draw.aaline(sc, BLACK, (tag_param*2, robot_rect[1] + 30), (robot_rect[0] + 25, robot_rect[1] + 30))
    pos_ll = text_ll.get_rect(center=((robot_rect[0] + 25 + tag_param*2) // 2,
                                       robot_rect[1] + 30))
    sc.blit(text_ll, pos_ll)

    # lr
    lr_x = W - tag_param*2 - 2
    pygame.draw.aaline(sc, BLACK, (W - tag_param*2 - 2, robot_rect[1] + 30), (robot_rect[0] + 25, robot_rect[1] + 30))
    pos_lr = text_lr.get_rect(center=((robot_rect[0] + 25 + W - tag_param*2 - 2) // 2,
                                       robot_rect[1] + 30))
    sc.blit(text_lr, pos_lr)




    x_robot, y_robot = getCoordsWithMatrix()
    sc_text = text.render(f'r1 = {r_math(x0, y0, robot_rect[0]+25, robot_rect[1]+30):.0f},'
                          f' r2 = {r_math(x1, y1, robot_rect[0]+25, robot_rect[1]+30):.0f},'
                          f' r3 = {r_math(x2, y2, robot_rect[0]+25, robot_rect[1]+30):.0f},'
                          f'lu = {(robot_rect[1] - tag_param + 30):.0f},'
                          f' ld = {(H - robot_rect[1] - 4*tag_param + 30):.0f},'
                          f' ll = {(robot_rect[0] - 2*tag_param + 25):.0f},'
                          f' ld = {(W - robot_rect[0] - 4*tag_param + 25):.0f},'
                          f' x_robot = {x_robot - tag_param*2:.0f},'
                          f' y_robot = {y_robot - tag_param:.0f}',
                          1, BLACK, WHITE)
    pos = sc_text.get_rect(center=(W // 2, H - tag_param*2 + 15))
    sc.blit(sc_text, pos)
    pygame.display.update()

    clock.tick(FPS)
