#Importar los paquetes necesarios
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import tkinter as tk
from kivy.app import App
#from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.core.image import Image
from kivy.graphics import BorderImage
from kivy.graphics import Color, Rectangle
from kivy.uix.image import AsyncImage
from kivy.uix.carousel import Carousel
from kivy.effects.kinetic import KineticEffect
from kivy.base import runTouchApp
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.lang import Builder
#from backend_kivy import FigureCanvasKivy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#Lectura del dataset
ds = pd.read_csv('google_review_ratings.csv')

#Eliminar columna vacía
ds = ds.drop(ds.columns[[25]], axis='columns')

#Quitar NAs
ds = ds.dropna(how='all')

#Renombrar columnas por su categoría
ds = ds.rename(index=str, columns={"Category 1": "Churches", "Category 2": "Resorts", "Category 3": "Beaches", "Category 4": "Parks", "Category 5": "Theatres"
	, "Category 6": "Museums", "Category 7": "Malls", "Category 8": "Zoo", "Category 9": "Restaurants", "Category 10": "Pubs/Bars", "Category 11": "Local services", "Category 12": "Burger/Pizza shops"
	, "Category 13": "Hotels/Other lodgings", "Category 14": "Juice bars", "Category 15": "Art galleries", "Category 16": "Dance clubs", "Category 17": "Swimming pools", "Category 18": "Gyms", "Category 19": "Bakeries"
	, "Category 20": "Beauty & Spas", "Category 21": "Cafes", "Category 22": "View points", "Category 23": "Monuments", "Category 24": "Gardens"})

#Guardar número total de usuarios
num_usuarios = len(ds.index)

# #Clustering

pd.options.display.max_columns = None
pd.options.display.max_rows = None
#Eliminamos la columna usuario
ds_norm = ds.drop(['User'], axis=1)

#Normalizacion de los datos
ds_norm = (ds_norm-ds_norm.min())/(ds_norm.max()-ds_norm.min())


# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
# #Eliminamos la columna usuario
# ds_norm = ds.drop(['User'], axis=1)

# #Normalizacion de los datos
# ds_norm = (ds_norm-ds_norm.min())/(ds_norm.max()-ds_norm.min())

# #Codo de jambu
# #Lista para almacenar valores
# wcss = []

# for i in range(1,11):
# 	kmeans = KMeans(n_clusters = i, max_iter = 300)
# 	kmeans.fit(ds_norm)
# 	wcss.append(kmeans.inertia_)

# plt.plot(range(1,11), wcss)
# plt.title("Codo de Jambu")
# plt.xlabel('Numero de clusters')
# plt.ylabel('WCSS')
# #plt.show()

# clustering = KMeans(n_clusters = 4, max_iter = 300)
# clustering.fit(ds_norm)


# ds['Resultados'] = clustering.labels_

# #Visualizar los resultados del cluster
# pca = PCA(n_components=2)
# pca_ds = pca.fit_transform(ds_norm)
# pca_ds_df = pd.DataFrame(data = pca_ds, columns = ['Componente_1', 'Componente_2'])
# pca_nombres_ds = pd.concat([pca_ds_df.reset_index(drop=True), ds['Resultados'].reset_index(drop=True)], axis = 1)
# print(pca_nombres_ds)


# #print(pca_nombres_ds)

# fig = plt.figure(figsize = (7,7))

# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('X', fontsize = 12)
# ax.set_ylabel('Y', fontsize = 12)
# ax.set_title('Clustering de usuarios', fontsize = 18)

# color_theme = np.array(["blue", "green", "orange","red"])
# ax.scatter(x = pca_nombres_ds.Componente_1, y = pca_nombres_ds.Componente_2,
# 			c=color_theme[pca_nombres_ds.Resultados], s=50)
# plt.show()

# #Eliminar columna de resultados para graficos info
# ds = ds.drop(ds.columns[[25]], axis='columns')

class StartScreen(Screen):
	pass

class IntroScreen(Screen):
	pass


class RecoScreen(Screen):

	#Clustering

	def codo(self):

		#Codo de jambu
		#Lista para almacenar valores
		wcss = []

		for i in range(1,11):
			kmeans = KMeans(n_clusters = i, max_iter = 300)
			kmeans.fit(ds_norm)
			wcss.append(kmeans.inertia_)

		fig = plt.figure(figsize=(7, 5))
		fig.canvas.set_window_title('Codo de Jambú')
		plt.plot(range(1,11), wcss)
		plt.title("Codo de Jambú")
		plt.xlabel('Número de clusters')
		plt.ylabel('WCSS')
		plt.show()

	def cluster(self):

		global ds
		global gr0 
		global gr1 
		global gr2 
		global gr3

		clustering = KMeans(n_clusters = 4, max_iter = 300)
		clustering.fit(ds_norm)


		ds['Resultados'] = clustering.labels_

		#Visualizar los resultados del cluster
		pca = PCA(n_components=2)
		pca_ds = pca.fit_transform(ds_norm)
		pca_ds_df = pd.DataFrame(data = pca_ds, columns = ['Componente_1', 'Componente_2'])
		pca_nombres_ds = pd.concat([pca_ds_df.reset_index(drop=True), ds['Resultados'].reset_index(drop=True)], axis = 1)
		#print(pca_nombres_ds)


		#print(pca_nombres_ds)

		fig = plt.figure(figsize = (7,7))
		fig.canvas.set_window_title('Cluster')

		ax = fig.add_subplot(1,1,1)
		ax.set_xlabel('X', fontsize = 12)
		ax.set_ylabel('Y', fontsize = 12)
		ax.set_title('Clustering de usuarios', fontsize = 18)

		color_theme = np.array(["blue", "green", "orange","red"])
		ax.scatter(x = pca_nombres_ds.Componente_1, y = pca_nombres_ds.Componente_2,
			c=color_theme[pca_nombres_ds.Resultados], s=50)

		gr0 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 0].tolist()].mean(axis=0)
		gr1 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 1].tolist()].mean(axis=0)
		gr2 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 2].tolist()].mean(axis=0)
		gr3 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 3].tolist()].mean(axis=0)
		plt.show()

		#Eliminar columna de resultados para graficos info
		ds = ds.drop(ds.columns[[25]], axis='columns')

class FinalRecoScreen(Screen):

		def press(self):

			# #Create variables for widget
			# name = self.ids.name_input.text
			# #Update label
			# self.ids.name_label.text = name
			# #Clear input box
			# self.ids.name_input.text = ''

			nota1 = self.ids.name_input1.text
			nota2 = self.ids.name_input2.text
			nota3 = self.ids.name_input3.text
			nota4 = self.ids.name_input4.text
			nota5 = self.ids.name_input5.text
			nota6 = self.ids.name_input6.text
			nota7 = self.ids.name_input7.text
			nota8 = self.ids.name_input8.text
			nota9 = self.ids.name_input9.text
			nota10 = self.ids.name_input10.text
			nota11 = self.ids.name_input11.text
			nota12 = self.ids.name_input12.text
			nota13 = self.ids.name_input13.text
			nota14 = self.ids.name_input14.text
			nota15 = self.ids.name_input15.text
			nota16 = self.ids.name_input16.text
			nota17 = self.ids.name_input17.text
			nota18 = self.ids.name_input18.text
			nota19 = self.ids.name_input19.text
			nota20 = self.ids.name_input20.text
			nota21 = self.ids.name_input21.text
			nota22 = self.ids.name_input22.text
			nota23 = self.ids.name_input23.text
			nota24 = self.ids.name_input24.text

			if (nota1 != '' and nota2 != '' and nota3 != '' and nota4 != '' and nota5 != '' and nota6 != '' and nota7 != '' and nota8 != '' and nota9 != ''
				 and nota10 != '' and nota11 != '' and nota12 != '' and nota13 != '' and nota14 != '' and nota15 != '' and nota16 != '' and nota17 != '' and nota18 != ''
				  and nota19 != '' and nota20 != '' and nota21 != '' and nota22 != '' and nota23 != '' and nota24 != ''):

				nota1 = float(nota1)
				nota2 = float(nota2)
				nota3 = float(nota3)
				nota4 = float(nota4)
				nota5 = float(nota5)
				nota6 = float(nota6)
				nota7 = float(nota7)
				nota8 = float(nota8)
				nota9 = float(nota9)
				nota10 = float(nota10)
				nota11 = float(nota11)
				nota12 = float(nota12)
				nota13 = float(nota13)
				nota14 = float(nota14)
				nota15 = float(nota15)
				nota16 = float(nota16)
				nota17 = float(nota17)
				nota18 = float(nota18)
				nota19 = float(nota19)
				nota20 = float(nota20)
				nota21 = float(nota21)
				nota22 = float(nota22)
				nota23 = float(nota23)
				nota24 = float(nota24)
				

				if (0 <= nota1 <= 5 and 0 <= nota2 <= 5 and 0 <= nota3 <= 5 and 0 <= nota4 <= 5 and 0 <= nota5 <= 5 and 0 <= nota6 <= 5 and 0 <= nota7 <= 5
				and 0 <= nota8 <= 5 and 0 <= nota9 <= 5 and 0 <= nota10 <= 5 and 0 <= nota11 <= 5 and 0 <= nota12 <= 5 and 0 <= nota13 <= 5 and 0 <= nota14 <= 5
				and 0 <= nota15 <= 5 and 0 <= nota16 <= 5 and 0 <= nota17 <= 5 and 0 <= nota18 <= 5 and 0 <= nota19 <= 5 and 0 <= nota20 <= 5 and 0 <= nota21 <= 5
				and 0 <= nota22 <= 5 and 0 <= nota23 <= 5 and 0 <= nota24 <= 5):


					self.ids.name_input1.text = ''
					self.ids.name_input2.text = ''
					self.ids.name_input3.text = ''
					self.ids.name_input4.text = ''
					self.ids.name_input5.text = ''
					self.ids.name_input6.text = ''
					self.ids.name_input7.text = ''
					self.ids.name_input8.text = ''
					self.ids.name_input9.text = ''
					self.ids.name_input10.text = ''
					self.ids.name_input11.text = ''
					self.ids.name_input12.text = ''
					self.ids.name_input13.text = ''
					self.ids.name_input14.text = ''
					self.ids.name_input15.text = ''
					self.ids.name_input16.text = ''
					self.ids.name_input17.text = ''
					self.ids.name_input18.text = ''
					self.ids.name_input19.text = ''
					self.ids.name_input20.text = ''
					self.ids.name_input21.text = ''
					self.ids.name_input22.text = ''
					self.ids.name_input23.text = ''
					self.ids.name_input24.text = ''

			
					global ds

					nuevo_usuario = {'User': num_usuarios+1, 'Churches':nota8, 'Resorts':nota18, 'Beaches':nota16, 'Parks': nota14, 'Theatres': nota22, 'Museums': nota12, 'Malls': nota2, 'Zoo': nota23,
					'Restaurants': nota19, 'Pubs/Bars': nota17, 'Local services': nota20, 'Burger/Pizza shops': nota6, 'Hotels/Other lodgings': nota7, 'Juice bars': nota24, 'Art galleries': nota4,
					'Dance clubs': nota3, 'Swimming pools': nota15, 'Gyms': nota5, 'Bakeries': nota13, 'Beauty & Spas': nota21, 'Cafes': nota1, 'View points': nota10, 'Monuments': nota11, 'Gardens': nota9}

					ds = ds.append(nuevo_usuario, ignore_index=True)

					self.ids.name_label_resultado.text = "Usuario añadido"

					ds_norm = ds.drop(['User'], axis=1)

					#Normalizacion de los datos
					ds_norm = (ds_norm-ds_norm.min())/(ds_norm.max()-ds_norm.min())

					clustering = KMeans(n_clusters = 4, max_iter = 300)
					clustering.fit(ds_norm)


					ds['Resultados'] = clustering.labels_

					#Visualizar los resultados del cluster
					pca = PCA(n_components=2)
					pca_ds = pca.fit_transform(ds_norm)
					pca_ds_df = pd.DataFrame(data = pca_ds, columns = ['Componente_1', 'Componente_2'])
					pca_nombres_ds = pd.concat([pca_ds_df.reset_index(drop=True), ds['Resultados'].reset_index(drop=True)], axis = 1)
					#print(pca_nombres_ds)


					#print(pca_nombres_ds)

					fig = plt.figure(figsize = (7,7))
					fig.canvas.set_window_title('Cluster')

					ax = fig.add_subplot(1,1,1)
					ax.set_xlabel('X', fontsize = 12)
					ax.set_ylabel('Y', fontsize = 12)
					ax.set_title('Clustering de usuarios', fontsize = 18)

					color_theme = np.array(["blue", "green", "orange","red"])
					ax.scatter(x = pca_nombres_ds.Componente_1, y = pca_nombres_ds.Componente_2,
						c=color_theme[pca_nombres_ds.Resultados], s=50)

					grupousu = ds.loc[5456, "Resultados"]
					grupousu = str(grupousu)
					self.ids.name_label_resultado.text = "Grupo "+ grupousu 

					#Eliminar columna de resultados para graficos info
					ds = ds.drop(ds.columns[[25]], axis='columns')
					

				else:
					self.ids.name_label_resultado.text = "Datos erroneos"

			else:
				self.ids.name_label_resultado.text = "Datos erroneos"


		# clustering = KMeans(n_clusters = 4, max_iter = 300)
		# clustering.fit(ds_norm)


		# ds['Resultados'] = clustering.labels_

		# #Visualizar los resultados del cluster
		# pca = PCA(n_components=2)
		# pca_ds = pca.fit_transform(ds_norm)
		# pca_ds_df = pd.DataFrame(data = pca_ds, columns = ['Componente_1', 'Componente_2'])
		# pca_nombres_ds = pd.concat([pca_ds_df.reset_index(drop=True), ds['Resultados'].reset_index(drop=True)], axis = 1)
		# print(pca_nombres_ds)


		# #print(pca_nombres_ds)

		# fig = plt.figure(figsize = (7,7))

		# ax = fig.add_subplot(1,1,1)
		# ax.set_xlabel('X', fontsize = 12)
		# ax.set_ylabel('Y', fontsize = 12)
		# ax.set_title('Clustering de usuarios', fontsize = 18)

		# color_theme = np.array(["blue", "green", "orange","red"])
		# ax.scatter(x = pca_nombres_ds.Componente_1, y = pca_nombres_ds.Componente_2,
		# 	c=color_theme[pca_nombres_ds.Resultados], s=50)
		# plt.show()

		# #Eliminar columna de resultados para graficos info
		# ds = ds.drop(ds.columns[[25]], axis='columns')

class EndScreen(Screen):

		clustering = KMeans(n_clusters = 4, max_iter = 300)
		clustering.fit(ds_norm)


		ds['Resultados'] = clustering.labels_

		#Visualizar los resultados del cluster
		pca = PCA(n_components=2)
		pca_ds = pca.fit_transform(ds_norm)
		pca_ds_df = pd.DataFrame(data = pca_ds, columns = ['Componente_1', 'Componente_2'])
		pca_nombres_ds = pd.concat([pca_ds_df.reset_index(drop=True), ds['Resultados'].reset_index(drop=True)], axis = 1)
		#print(pca_nombres_ds)
		gr0 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 0].tolist()].mean(axis=0)
		gr1 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 1].tolist()].mean(axis=0)
		gr2 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 2].tolist()].mean(axis=0)
		gr3 = ds_norm.iloc[pca_nombres_ds.index[pca_nombres_ds['Resultados'] == 3].tolist()].mean(axis=0)

		def grupo0(self):

			df0 = pd.DataFrame(gr0, columns = ['Media'])
			categoria = ['Iglesias', 'Resorts', 'Playas', 'Parques', 'Teatros', 'Museos', 'C.Comerciales', 'Zoos', 'Restaurantes',
			'Pubs/Bares', 'Servicios locales', 'Hamburgueserías', 'Hoteles', 'Zumerías', 'Galerías de arte', 'Discotecas', 'Piscinas', 'Gimnasios',
			'Panaderías', 'Spas', 'Cafeterías', 'Miradores', 'Monumentos', 'Jardines públicos']
			df0['Categoría'] = categoria
			fig = plt.figure(figsize = (18,9))
			fig.canvas.set_window_title('Grupo 0')
			plt.plot(df0['Categoría'],df0['Media'], 'og--')
			plt.title("Recomendación basada en las valoraciones de los usuarios del grupo 0", 
          	fontdict={'family': 'serif', 
                    'color' : 'darkblue',
                    'weight': 'bold',
                    'size': 20})
			plt.xticks(rotation=35)
			plt.show()

		def grupo1(self):

			df1 = pd.DataFrame(gr1, columns = ['Media'])
			categoria = ['Iglesias', 'Resorts', 'Playas', 'Parques', 'Teatros', 'Museos', 'C.Comerciales', 'Zoos', 'Restaurantes',
			'Pubs/Bares', 'Servicios locales', 'Hamburgueserías', 'Hoteles', 'Zumerías', 'Galerías de arte', 'Discotecas', 'Piscinas', 'Gimnasios',
			'Panaderías', 'Spas', 'Cafeterías', 'Miradores', 'Monumentos', 'Jardines públicos']
			df1['Categoría'] = categoria
			fig = plt.figure(figsize = (18,9))
			fig.canvas.set_window_title('Grupo 1')
			plt.plot(df1['Categoría'],df1['Media'], 'og--')
			plt.title("Recomendación basada en las valoraciones de los usuarios del grupo 1", 
          	fontdict={'family': 'serif', 
                    'color' : 'darkblue',
                    'weight': 'bold',
                    'size': 20})
			plt.xticks(rotation=35)
			plt.show()

		def grupo2(self):

			df2 = pd.DataFrame(gr2, columns = ['Media'])
			categoria = ['Iglesias', 'Resorts', 'Playas', 'Parques', 'Teatros', 'Museos', 'C.Comerciales', 'Zoos', 'Restaurantes',
			'Pubs/Bares', 'Servicios locales', 'Hamburgueserías', 'Hoteles', 'Zumerías', 'Galerías de arte', 'Discotecas', 'Piscinas', 'Gimnasios',
			'Panaderías', 'Spas', 'Cafeterías', 'Miradores', 'Monumentos', 'Jardines públicos']
			df2['Categoría'] = categoria
			fig = plt.figure(figsize = (18,9))
			fig.canvas.set_window_title('Grupo 2')
			plt.plot(df2['Categoría'],df2['Media'], 'og--')
			plt.title("Recomendación basada en las valoraciones de los usuarios del grupo 2", 
          	fontdict={'family': 'serif', 
                    'color' : 'darkblue',
                    'weight': 'bold',
                    'size': 20})
			plt.xticks(rotation=35)
			plt.show()

		def grupo3(self):

			df3 = pd.DataFrame(gr3, columns = ['Media'])
			categoria = ['Iglesias', 'Resorts', 'Playas', 'Parques', 'Teatros', 'Museos', 'C.Comerciales', 'Zoos', 'Restaurantes',
			'Pubs/Bares', 'Servicios locales', 'Hamburgueserías', 'Hoteles', 'Zumerías', 'Galerías de arte', 'Discotecas', 'Piscinas', 'Gimnasios',
			'Panaderías', 'Spas', 'Cafeterías', 'Miradores', 'Monumentos', 'Jardines públicos']
			df3['Categoría'] = categoria
			fig = plt.figure(figsize = (18,9))
			fig.canvas.set_window_title('Grupo 3')
			plt.plot(df3['Categoría'],df3['Media'], 'og--')
			plt.title("Recomendación basada en las valoraciones de los usuarios del grupo 3", 
          	fontdict={'family': 'serif', 
                    'color' : 'darkblue',
                    'weight': 'bold',
                    'size': 20})
			plt.xticks(rotation=35)
			plt.show()


class InfoScreen(Screen):

	def iglesias(self):
		###PRIMER GRAFICO
		#Dejar solo la columna necesaria
		df1 = ds.drop(ds.columns[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden = df1.sort_values('Churches', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden.index:
			total = orden['Churches'][value] + total
		media = total/len(orden)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad1 = orden['Churches'].value_counts()

		#Ordenar de menor a mayor
		cantidad1 = cantidad1.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf1 = pd.DataFrame(cantidad1)
		graf1_reset = graf1.reset_index()
		graf1_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Iglesias')
		plt.style.use('seaborn')
		plt.scatter(graf1_reset.Values, graf1_reset.Quantities, c=graf1_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE IGLESIAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		# def __init__(self,**kwargs):
		#  	super(graf1,self).__init__(**kwargs)
		#  	box = self.ids.igle_screen.ids.box
		#  	box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		#  	return box

		# def __init__(self,**kwargs):
		#  	super().__init__(**kwargs)
		# 	box = self.ids.box
		#  	box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		#Asegurar que en el gráfico salen la misma cantidad de notas que de reseñas
		# cuantos = 0
		# contador = 0
		# while contador < len(graf1_reset['Quantities']):
		# 	cuantos += graf1_reset['Quantities'][contador]
		# 	contador += 1

		#Visualizar dataset final
		#print(graf1_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf1_reset, x = 'Values', y = 'Quantities', trendline= 'lowess', trendline_options=dict(frac=0.1),  color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE IGLESIAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
  #                             	line=dict(width=2,
  #                                       	color='DarkSlateGray')),
  #                 	selector=dict(mode='markers'))
		# fig.show()

	# def build(self):
	# 	box = self.ids.box
	# 	box.add_widget(FigureCanvasKivyAgg(iglesias().gcf()))
	# 	return box


	def resorts(self):

		#Dejar solo la columna necesaria
		df2 = ds.drop(ds.columns[[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden2 = df2.sort_values('Resorts', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden2.index:
			total = orden2['Resorts'][value] + total
		media = total/len(orden2)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad2 = orden2['Resorts'].value_counts()

		#Ordenar de menor a mayor
		cantidad2 = cantidad2.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf2 = pd.DataFrame(cantidad2)
		graf2_reset = graf2.reset_index()
		graf2_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Resorts')
		plt.style.use('seaborn')
		plt.scatter(graf2_reset.Values, graf2_reset.Quantities, c=graf2_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE RESORTS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf2_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf2_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE RESORTS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def playas(self):

		#Dejar solo la columna necesaria
		df3 = ds.drop(ds.columns[[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden3 = df3.sort_values('Beaches', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden3.index:
			total = orden3['Beaches'][value] + total
		media = total/len(orden3)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad3 = orden3['Beaches'].value_counts()

		#Ordenar de menor a mayor
		cantidad3 = cantidad3.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf3 = pd.DataFrame(cantidad3)
		graf3_reset = graf3.reset_index()
		graf3_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Playas')
		plt.style.use('seaborn')
		plt.scatter(graf3_reset.Values, graf3_reset.Quantities, c=graf3_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE PLAYAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf3_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf3_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE PLAYAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def parques(self):

		#Dejar solo la columna necesaria
		df4 = ds.drop(ds.columns[[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden4 = df4.sort_values('Parks', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden4.index:
			total = orden4['Parks'][value] + total
		media = total/len(orden4)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad4 = orden4['Parks'].value_counts()

		#Ordenar de menor a mayor
		cantidad4 = cantidad4.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf4 = pd.DataFrame(cantidad4)
		graf4_reset = graf4.reset_index()
		graf4_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Parques')
		plt.style.use('seaborn')
		plt.scatter(graf4_reset.Values, graf4_reset.Quantities, c=graf4_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE PARQUES                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf4_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf4_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE PARQUES' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def teatros(self):

		#Dejar solo la columna necesaria
		df5 = ds.drop(ds.columns[[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden5 = df5.sort_values('Theatres', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden5.index:
			total = orden5['Theatres'][value] + total
		media = total/len(orden5)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad5 = orden5['Theatres'].value_counts()

		#Ordenar de menor a mayor
		cantidad5 = cantidad5.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf5 = pd.DataFrame(cantidad5)
		graf5_reset = graf5.reset_index()
		graf5_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Teatros')
		plt.style.use('seaborn')
		plt.scatter(graf5_reset.Values, graf5_reset.Quantities, c=graf5_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE TEATROS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf5_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf5_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE TEATROS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def museos(self):

		#Dejar solo la columna necesaria
		df6 = ds.drop(ds.columns[[1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden6 = df6.sort_values('Museums', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden6.index:
			total = orden6['Museums'][value] + total
		media = total/len(orden6)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad6 = orden6['Museums'].value_counts()

		#Ordenar de menor a mayor
		cantidad6 = cantidad6.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf6 = pd.DataFrame(cantidad6)
		graf6_reset = graf6.reset_index()
		graf6_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Museos')
		plt.style.use('seaborn')
		plt.scatter(graf6_reset.Values, graf6_reset.Quantities, c=graf6_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE MUSEOS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf6_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf6_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE MUSEOS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def ccomerciales(self):

		#Dejar solo la columna necesaria
		df7 = ds.drop(ds.columns[[1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden7 = df7.sort_values('Malls', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden7.index:
			total = orden7['Malls'][value] + total
		media = total/len(orden7)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad7 = orden7['Malls'].value_counts()

		#Ordenar de menor a mayor
		cantidad7 = cantidad7.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf7 = pd.DataFrame(cantidad7)
		graf7_reset = graf7.reset_index()
		graf7_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Centros comerciales')
		plt.style.use('seaborn')
		plt.scatter(graf7_reset.Values, graf7_reset.Quantities, c=graf7_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE CENTROS COMERCIALES                PUNTUACIÓN MEDIA: " + media)
		plt.show()



		#Visualizar dataset final
		#print(graf7_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf7_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE CENTROS COMERCIALES' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def zoos(self):

		#Dejar solo la columna necesaria
		df8 = ds.drop(ds.columns[[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden8 = df8.sort_values('Zoo', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden8.index:
			total = orden8['Zoo'][value] + total
		media = total/len(orden8)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad8 = orden8['Zoo'].value_counts()

		#Ordenar de menor a mayor
		cantidad8 = cantidad8.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf8 = pd.DataFrame(cantidad8)
		graf8_reset = graf8.reset_index()
		graf8_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Zoos')
		plt.style.use('seaborn')
		plt.scatter(graf8_reset.Values, graf8_reset.Quantities, c=graf8_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE ZOOS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf8_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf8_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE ZOOS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def restaurantes(self):

		#Dejar solo la columna necesaria
		df9 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden9 = df9.sort_values('Restaurants', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden9.index:
			total = orden9['Restaurants'][value] + total
		media = total/len(orden9)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad9 = orden9['Restaurants'].value_counts()

		#Ordenar de menor a mayor
		cantidad9 = cantidad9.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf9 = pd.DataFrame(cantidad9)
		graf9_reset = graf9.reset_index()
		graf9_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Restaurantes')
		plt.style.use('seaborn')
		plt.scatter(graf9_reset.Values, graf9_reset.Quantities, c=graf9_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE RESTAURANTES                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf9_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf9_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE RESTAURANTES' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def pubsbars(self):

		#Dejar solo la columna necesaria
		df10 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden10 = df10.sort_values('Pubs/Bars', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden10.index:
			total = orden10['Pubs/Bars'][value] + total
		media = total/len(orden10)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad10 = orden10['Pubs/Bars'].value_counts()

		#Ordenar de menor a mayor
		cantidad10 = cantidad10.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf10 = pd.DataFrame(cantidad10)
		graf10_reset = graf10.reset_index()
		graf10_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Pubs/Bares')
		plt.style.use('seaborn')
		plt.scatter(graf10_reset.Values, graf10_reset.Quantities, c=graf10_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE PUBS/BARES                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf10_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf10_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE PUBS/BARES' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def servicioslocales(self):

		#Dejar solo la columna necesaria
		df11 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden11 = df11.sort_values('Local services', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden11.index:
			total = orden11['Local services'][value] + total
		media = total/len(orden11)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad11 = orden11['Local services'].value_counts()

		#Ordenar de menor a mayor
		cantidad11 = cantidad11.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf11 = pd.DataFrame(cantidad11)
		graf11_reset = graf11.reset_index()
		graf11_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Servicios locales')
		plt.style.use('seaborn')
		plt.scatter(graf11_reset.Values, graf11_reset.Quantities, c=graf11_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE SERVICIOS LOCALES                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf11_reset)


		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf11_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE SERVICIOS LOCALES' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def burgerpizza(self):

		#Dejar solo la columna necesaria
		df12 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden12 = df12.sort_values('Burger/Pizza shops', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden12.index:
			total = orden12['Burger/Pizza shops'][value] + total
		media = total/len(orden12)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad12 = orden12['Burger/Pizza shops'].value_counts()

		#Ordenar de menor a mayor
		cantidad12 = cantidad12.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf12 = pd.DataFrame(cantidad12)
		graf12_reset = graf12.reset_index()
		graf12_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Hamburgueserías/Pizzerías')
		plt.style.use('seaborn')
		plt.scatter(graf12_reset.Values, graf12_reset.Quantities, c=graf12_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE HAMBURGUESERÍAS/PIZZERÍAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf12_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf12_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE HAMBURGUESERÍAS/PIZZERÍAS' + '\t'*175 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def hoteles(self):

		#Dejar solo la columna necesaria
		df13 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden13 = df13.sort_values('Hotels/Other lodgings', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden13.index:
			total = orden13['Hotels/Other lodgings'][value] + total
		media = total/len(orden13)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad13 = orden13['Hotels/Other lodgings'].value_counts()

		#Ordenar de menor a mayor
		cantidad13 = cantidad13.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf13 = pd.DataFrame(cantidad13)
		graf13_reset = graf13.reset_index()
		graf13_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Hoteles')
		plt.style.use('seaborn')
		plt.scatter(graf13_reset.Values, graf13_reset.Quantities, c=graf13_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE HOTELES               PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf13_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf13_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE HOTELES' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()


	def zumos(self):

		#Dejar solo la columna necesaria
		df14 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden14 = df14.sort_values('Juice bars', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden14.index:
			total = orden14['Juice bars'][value] + total
		media = total/len(orden14)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad14 = orden14['Juice bars'].value_counts()

		#Ordenar de menor a mayor
		cantidad14 = cantidad14.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf14 = pd.DataFrame(cantidad14)
		graf14_reset = graf14.reset_index()
		graf14_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Zumerías')
		plt.style.use('seaborn')
		plt.scatter(graf14_reset.Values, graf14_reset.Quantities, c=graf14_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE ZUMERÍAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf14_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf14_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE ZUMERÍAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def galerias(self):

		#Dejar solo la columna necesaria
		df15 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden15 = df15.sort_values('Art galleries', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden15.index:
			total = orden15['Art galleries'][value] + total
		media = total/len(orden15)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad15 = orden15['Art galleries'].value_counts()

		#Ordenar de menor a mayor
		cantidad15 = cantidad15.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf15 = pd.DataFrame(cantidad15)
		graf15_reset = graf15.reset_index()
		graf15_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Galerías de arte')
		plt.style.use('seaborn')
		plt.scatter(graf15_reset.Values, graf15_reset.Quantities, c=graf15_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE GALERÍAS DE ARTE                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf15_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf15_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE GALERÍAS DE ARTE' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def discos(self):

		#Dejar solo la columna necesaria
		df16 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden16 = df16.sort_values('Dance clubs', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden16.index:
			total = orden16['Dance clubs'][value] + total
		media = total/len(orden16)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad16 = orden16['Dance clubs'].value_counts()

		#Ordenar de menor a mayor
		cantidad16 = cantidad16.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf16 = pd.DataFrame(cantidad16)
		graf16_reset = graf16.reset_index()
		graf16_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Discotecas')
		plt.style.use('seaborn')
		plt.scatter(graf16_reset.Values, graf16_reset.Quantities, c=graf16_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE DISCOTECAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf16_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf16_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE DISCOTECAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def piscinas(self):

		#Dejar solo la columna necesaria
		df17 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden17 = df17.sort_values('Swimming pools', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden17.index:
			total = orden17['Swimming pools'][value] + total
		media = total/len(orden17)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad17 = orden17['Swimming pools'].value_counts()

		#Ordenar de menor a mayor
		cantidad17 = cantidad17.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf17 = pd.DataFrame(cantidad17)
		graf17_reset = graf17.reset_index()
		graf17_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Piscinas')
		plt.style.use('seaborn')
		plt.scatter(graf17_reset.Values, graf17_reset.Quantities, c=graf17_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE PISCINAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf17_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf17_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE PISCINAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def gimnasios(self):

		#Dejar solo la columna necesaria
		df18 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden18 = df18.sort_values('Gyms', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden18.index:
			total = orden18['Gyms'][value] + total
		media = total/len(orden18)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad18 = orden18['Gyms'].value_counts()

		#Ordenar de menor a mayor
		cantidad18 = cantidad18.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf18 = pd.DataFrame(cantidad18)
		graf18_reset = graf18.reset_index()
		graf18_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Gimnasios')
		plt.style.use('seaborn')
		plt.scatter(graf18_reset.Values, graf18_reset.Quantities, c=graf18_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE GIMNASIOS               PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf18_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf18_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE GIMNASIOS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def panaderias(self):

		#Dejar solo la columna necesaria
		df19 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden19 = df19.sort_values('Bakeries', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden19.index:
			total = orden19['Bakeries'][value] + total
		media = total/len(orden19)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad19 = orden19['Bakeries'].value_counts()

		#Ordenar de menor a mayor
		cantidad19 = cantidad19.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf19 = pd.DataFrame(cantidad19)
		graf19_reset = graf19.reset_index()
		graf19_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Panaderías')
		plt.style.use('seaborn')
		plt.scatter(graf19_reset.Values, graf19_reset.Quantities, c=graf19_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE PANADERÍAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf19_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf19_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE PANADERÍAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def spas(self):

		#Dejar solo la columna necesaria
		df20 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden20 = df20.sort_values('Beauty & Spas', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden20.index:
			total = orden20['Beauty & Spas'][value] + total
		media = total/len(orden20)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad20 = orden20['Beauty & Spas'].value_counts()

		#Ordenar de menor a mayor
		cantidad20 = cantidad20.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf20 = pd.DataFrame(cantidad20)
		graf20_reset = graf20.reset_index()
		graf20_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Spas')
		plt.style.use('seaborn')
		plt.scatter(graf20_reset.Values, graf20_reset.Quantities, c=graf20_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE SPAS               PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf20_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf20_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE SPAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def cafes(self):

		#Dejar solo la columna necesaria
		df21 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden21 = df21.sort_values('Cafes', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden21.index:
			total = orden21['Cafes'][value] + total
		media = total/len(orden21)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad21 = orden21['Cafes'].value_counts()

		#Ordenar de menor a mayor
		cantidad21 = cantidad21.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf21 = pd.DataFrame(cantidad21)
		graf21_reset = graf21.reset_index()
		graf21_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Cafeterías')
		plt.style.use('seaborn')
		plt.scatter(graf21_reset.Values, graf21_reset.Quantities, c=graf21_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE CAFETERÍAS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf21_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf21_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE CAFETERÍAS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def miradores(self):

		#Dejar solo la columna necesaria
		df22 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden22 = df22.sort_values('View points', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden22.index:
			total = orden22['View points'][value] + total
		media = total/len(orden22)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad22 = orden22['View points'].value_counts()

		#Ordenar de menor a mayor
		cantidad22 = cantidad22.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf22 = pd.DataFrame(cantidad22)
		graf22_reset = graf22.reset_index()
		graf22_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Miradores')
		plt.style.use('seaborn')
		plt.scatter(graf22_reset.Values, graf22_reset.Quantities, c=graf22_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE MIRADORES               PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf22_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf22_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE MIRADORES' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def monumentos(self):

		#Dejar solo la columna necesaria
		df23 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24]], axis='columns')

		#Ordenar valores del 0 al 5
		orden23 = df23.sort_values('Monuments', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden23.index:
			total = orden23['Monuments'][value] + total
		media = total/len(orden23)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))

		#Cuantas veces sale cada nota
		cantidad23 = orden23['Monuments'].value_counts()

		#Ordenar de menor a mayor
		cantidad23 = cantidad23.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf23 = pd.DataFrame(cantidad23)
		graf23_reset = graf23.reset_index()
		graf23_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Monumentos')
		plt.style.use('seaborn')
		plt.scatter(graf23_reset.Values, graf23_reset.Quantities, c=graf23_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE MONUMENTOS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf23_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf23_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn,
		# title = 'REVIEWS DE MONUMENTOS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))

		# fig.show()

	def jardines(self):

		#Dejar solo la columna necesaria
		df24 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]], axis='columns')

		#Ordenar valores del 0 al 5
		orden24 = df24.sort_values('Gardens', ascending=True)

		#Calcular nota media de las reviews
		total = 0
		for value in orden24.index:
			total = orden24['Gardens'][value] + total
		media = total/len(orden24)
		media = "{:.2f}".format(media)
		#print("{:.2f}".format(media))


		#Cuantas veces sale cada nota
		cantidad24 = orden24['Gardens'].value_counts()

		#Ordenar de menor a mayor
		cantidad24 = cantidad24.sort_index()

		#Crear un nuevo dataframe con valores y las veces que sale cada valor
		graf24 = pd.DataFrame(cantidad24)
		graf24_reset = graf24.reset_index()
		graf24_reset.columns = ['Values', 'Quantities']

		fig = plt.figure(figsize=(10, 6))
		fig.canvas.set_window_title('Jardines públicos')
		plt.style.use('seaborn')
		plt.scatter(graf24_reset.Values, graf24_reset.Quantities, c=graf24_reset.Values, cmap='RdYlGn', edgecolor='black')
		plt.colorbar()
		plt.ylabel("Cantidad")
		plt.xlabel("Notas")
		plt.title("REVIEWS DE JARDINES PÚBLICOS                PUNTUACIÓN MEDIA: " + media)
		plt.show()

		#Visualizar dataset final
		#print(graf24_reset)

		#Crear gráfico a partir de los datos
		# fig = px.scatter(graf24_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, 
		# 	title = 'REVIEWS DE JARDINES PÚBLICOS' + '\t'*190 + 'PUNTUACIÓN MEDIA: ' + str(media))
		# fig.update_traces(marker=dict(size=12,
	 #                              line=dict(width=2,
	 #                                        color='DarkSlateGray')),
	 #                  selector=dict(mode='markers'))
		# fig.show()


#class RootScreen(ScreenManager):
#	pass

class TuristappApp(App):

	def change_screen(self, screen_name):
		self.root.current = screen_name

	# def showgraf(self):
	# 	#self.theme_cls.theme_style = "Dark"
	# 	#self.theme_cls.primary_palette = "BlueGray"
	# 	Builder.load_file('iglescreen.kv')
	# 	return graf1()

	# def build(self):
	# 	return RootScreen()



TuristappApp().run()
