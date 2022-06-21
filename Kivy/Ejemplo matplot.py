from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import plotly.express as px

x = [1,2,3,4,5]
y = [5,12,6,9,15]

plt.style.use('seaborn')
plt.scatter(x,y, c=x, cmap='RdYlGn', edgecolor='k')
plt.colorbar()
plt.ylabel("Y Axis")
plt.xlabel("X Axis")



class Matty(FloatLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		box = self.ids.box
		box.add_widget(FigureCanvasKivyAgg(plt.gcf()))

	def save_it(self):
		pass

class MainApp(MDApp):
	def build(self):
		self.theme_cls.theme_style = "Dark"
		self.theme_cls.primary_palette = "BlueGray"
		Builder.load_file('matty.kv')
		return Matty()

MainApp().run()

<RootScreen>:
	transition: FadeTransition()
	IntroScreen:
	StartScreen: 
	InfoScreen:
	RecoScreen:
	IgleScreen: 

on_release: root.manager.current = 'start'