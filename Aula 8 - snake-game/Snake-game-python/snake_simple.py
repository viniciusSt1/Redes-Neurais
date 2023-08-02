
import pygame
from pygame.locals import * 
from random import randint
from sys import exit

def init_game(largura, altura):
    pygame.init()
    speed = 20
    # Fonte, tamanho,  bold, italic
    fonte = pygame.font.SysFont('gabriola', 30, True, True)  # pygame.font.get_fonts()

    tela = pygame.display.set_mode((largura,altura))
    pygame.display.set_caption('Deep Snake Game')
    clock = pygame.time.Clock()  #objeto para controlar o fps do jogo
    reload_game()

def on_grid_random(largura,altura):
    x = randint(20, largura - 20)
    y = randint(20, altura - 20)
    return x // 10 * 10, y // 10 * 10

def gain_snake(lista_snake):
    for point in list_snake: #point pontos na lista contando as posições do corpo da cobra
        pygame.draw.rect(tela,(255,150,255),(point[0], point[1], 20, 20))

def reload_game():
    global pontos, comprimento_inicial, x, y, list_head, list_snake, x_food, y_food, dead
    pontos = 0
    comprimento_inicial = 1 
    x = (largura/2) // 20.0 * 10
    y = (altura/2) // 20.0 * 10
    list_head = []
    list_snake = []
    x_food = randint(40, 600)
    x_food = round(x_food / 20.0) * 10
    y_food = randint(40, 420)
    y_food = round(y_food / 20.0) * 10
    dead = False

pygame.init()
largura = 640
altura = 480
        
pontos = 0
comprimento_inicial = 1 
x = (largura/2) // 20.0 * 10
y = (altura/2) // 20.0 * 10
list_head = []
list_snake = []
x_food = randint(40, 600)
x_food = round(x_food / 20.0) * 10
y_food = randint(40, 420)
y_food = round(y_food / 20.0) * 10
dead = False    
    


speed  = 20
x_control = speed
y_control = 0

# Fonte, tamanho,  bold, italic
fonte = pygame.font.SysFont('gabriola', 30, True, True)  # pygame.font.get_fonts()

tela = pygame.display.set_mode((largura,altura))
pygame.display.set_caption('Deep Snake Game')
clock = pygame.time.Clock()  #objeto para controlar o fps do jogo

while True:
    clock.tick(10)
    tela.fill((50,50,50))
    mensagem = f'Pontos: {pontos}'
    texto = fonte.render(mensagem, True, (255,150,255))
    for event in pygame.event.get(): # loop que controla os eventos dentro do jogo
        if event.type == QUIT:
            pygame.quit()
            exit()   
        if event.type == KEYDOWN:    
            if event.key == K_UP:
                if y_control == speed:
                    pass
                else:
                    x_control = 0
                    y_control -= speed 
            if event.key == K_RIGHT:
                if x_control == -speed:
                    pass
                else:
                    x_control += speed  
                    y_control = 0
            if event.key == K_LEFT:
                if x_control == speed:
                    pass
                else:
                    x_control -= speed 
                    y_control = 0
            if event.key == K_DOWN:
                if y_control == -speed:
                    pass
                else:
                    x_control = 0
                    y_control += speed     
    # if pygame.key.get_pressed()[K_UP]:
    #     y -= 20 
    # if pygame.key.get_pressed()[K_RIGHT]:
    #     x += 20 
    # if pygame.key.get_pressed()[K_LEFT]:
    #     x -= 20 
    # if pygame.key.get_pressed()[K_DOWN]:
    #     y += 20    
    x += x_control
    y += y_control
    
    snake = pygame.draw.rect(tela, (255,150,255), (x,y,20,20))
    food = pygame.draw.rect(tela, (255,0,0), (x_food,y_food,20,20))
    
    if snake.colliderect(food):
        x_food, y_food = on_grid_random(largura,altura)
        pontos += 1
        comprimento_inicial += 1
        
    list_head = []
    list_head.append(x)
    list_head.append(y)
    list_snake.append(list_head)

    if list_snake.count(list_head) > 1:
        fonte2 = pygame.font.SysFont('gabriola', 30, True, True)
        mensagem = f'To reload press r: '
        texto = fonte2.render(mensagem, True, (255,150,255))
        ret_texto = texto.get_rect()
        
        dead = True
        while dead:
            tela.fill((255,255,255))
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()
                if event.type == KEYDOWN:
                    if event.key == K_r:
                        reload_game()
            ret_texto.center = (largura // 2, altura // 2)
            tela.blit(texto, ret_texto)
            pygame.display.update()               

    if x > largura:
        x = 0
    if x < 0:
        x = largura
    if y < 0:
        y = altura
    if y > altura:
        y = 0

    if len(list_snake) > comprimento_inicial:
        del list_snake[0]
    
    gain_snake(list_snake)
    
    tela.blit(texto, (520,10))
        
    pygame.display.update()
