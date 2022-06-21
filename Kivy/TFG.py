#Importar los paquetes necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

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
num_reseñas = len(ds.index)


def iglesias():
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
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad1 = orden['Churches'].value_counts()

	#Ordenar de menor a mayor
	cantidad1 = cantidad1.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf1 = pd.DataFrame(cantidad1)
	graf1_reset = graf1.reset_index()
	graf1_reset.columns = ['Values', 'Quantities']

	#Asegurar que en el gráfico salen la misma cantidad de notas que de reseñas
	# cuantos = 0
	# contador = 0
	# while contador < len(graf1_reset['Quantities']):
	# 	cuantos += graf1_reset['Quantities'][contador]
	# 	contador += 1

	#Visualizar dataset final
	#print(graf1_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf1_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'CHURCHES REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def resorts():

	#Dejar solo la columna necesaria
	df2 = ds.drop(ds.columns[[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden2 = df2.sort_values('Resorts', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden2.index:
		total = orden2['Resorts'][value] + total
	media = total/len(orden2)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad2 = orden2['Resorts'].value_counts()

	#Ordenar de menor a mayor
	cantidad2 = cantidad2.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf2 = pd.DataFrame(cantidad2)
	graf2_reset = graf2.reset_index()
	graf2_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf2_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf2_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'RESORTS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def playas():

	#Dejar solo la columna necesaria
	df3 = ds.drop(ds.columns[[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden3 = df3.sort_values('Beaches', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden3.index:
		total = orden3['Beaches'][value] + total
	media = total/len(orden3)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad3 = orden3['Beaches'].value_counts()

	#Ordenar de menor a mayor
	cantidad3 = cantidad3.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf3 = pd.DataFrame(cantidad3)
	graf3_reset = graf3.reset_index()
	graf3_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf3_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf3_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'BEACHES REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def parques():

	#Dejar solo la columna necesaria
	df4 = ds.drop(ds.columns[[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden4 = df4.sort_values('Parks', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden4.index:
		total = orden4['Parks'][value] + total
	media = total/len(orden4)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad4 = orden4['Parks'].value_counts()

	#Ordenar de menor a mayor
	cantidad4 = cantidad4.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf4 = pd.DataFrame(cantidad4)
	graf4_reset = graf4.reset_index()
	graf4_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf4_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf4_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'PARKS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def teatros():

	#Dejar solo la columna necesaria
	df5 = ds.drop(ds.columns[[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden5 = df5.sort_values('Theatres', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden5.index:
		total = orden5['Theatres'][value] + total
	media = total/len(orden5)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad5 = orden5['Theatres'].value_counts()

	#Ordenar de menor a mayor
	cantidad5 = cantidad5.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf5 = pd.DataFrame(cantidad5)
	graf5_reset = graf5.reset_index()
	graf5_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf5_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf5_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'THEATRES REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def museos():

	#Dejar solo la columna necesaria
	df6 = ds.drop(ds.columns[[1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden6 = df6.sort_values('Museums', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden6.index:
		total = orden6['Museums'][value] + total
	media = total/len(orden6)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad6 = orden6['Museums'].value_counts()

	#Ordenar de menor a mayor
	cantidad6 = cantidad6.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf6 = pd.DataFrame(cantidad6)
	graf6_reset = graf6.reset_index()
	graf6_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf6_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf6_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'MUSEUMS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def ccomerciales():

	#Dejar solo la columna necesaria
	df7 = ds.drop(ds.columns[[1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden7 = df7.sort_values('Malls', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden7.index:
		total = orden7['Malls'][value] + total
	media = total/len(orden7)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad7 = orden7['Malls'].value_counts()

	#Ordenar de menor a mayor
	cantidad7 = cantidad7.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf7 = pd.DataFrame(cantidad7)
	graf7_reset = graf7.reset_index()
	graf7_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf7_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf7_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'MALLS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def zoos():

	#Dejar solo la columna necesaria
	df8 = ds.drop(ds.columns[[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden8 = df8.sort_values('Zoo', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden8.index:
		total = orden8['Zoo'][value] + total
	media = total/len(orden8)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad8 = orden8['Zoo'].value_counts()

	#Ordenar de menor a mayor
	cantidad8 = cantidad8.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf8 = pd.DataFrame(cantidad8)
	graf8_reset = graf8.reset_index()
	graf8_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf8_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf8_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'ZOOS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def restaurantes():

	#Dejar solo la columna necesaria
	df9 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden9 = df9.sort_values('Restaurants', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden9.index:
		total = orden9['Restaurants'][value] + total
	media = total/len(orden9)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad9 = orden9['Restaurants'].value_counts()

	#Ordenar de menor a mayor
	cantidad9 = cantidad9.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf9 = pd.DataFrame(cantidad9)
	graf9_reset = graf9.reset_index()
	graf9_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf9_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf9_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'RESTAURANTS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def pubsbars():

	#Dejar solo la columna necesaria
	df10 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden10 = df10.sort_values('Pubs/Bars', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden10.index:
		total = orden10['Pubs/Bars'][value] + total
	media = total/len(orden10)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad10 = orden10['Pubs/Bars'].value_counts()

	#Ordenar de menor a mayor
	cantidad10 = cantidad10.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf10 = pd.DataFrame(cantidad10)
	graf10_reset = graf10.reset_index()
	graf10_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf10_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf10_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'PUBS/BARS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def servicioslocales():

	#Dejar solo la columna necesaria
	df11 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden11 = df11.sort_values('Local services', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden11.index:
		total = orden11['Local services'][value] + total
	media = total/len(orden11)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad11 = orden11['Local services'].value_counts()

	#Ordenar de menor a mayor
	cantidad11 = cantidad11.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf11 = pd.DataFrame(cantidad11)
	graf11_reset = graf11.reset_index()
	graf11_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf11_reset)


	#Crear gráfico a partir de los datos
	fig = px.scatter(graf11_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'LOCAL SERVICES REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def burgerpizza():

	#Dejar solo la columna necesaria
	df12 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden12 = df12.sort_values('Burger/Pizza shops', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden12.index:
		total = orden12['Burger/Pizza shops'][value] + total
	media = total/len(orden12)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad12 = orden12['Burger/Pizza shops'].value_counts()

	#Ordenar de menor a mayor
	cantidad12 = cantidad12.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf12 = pd.DataFrame(cantidad12)
	graf12_reset = graf12.reset_index()
	graf12_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf12_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf12_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'BURGER/PIZZA SHOPS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def hoteles():

	#Dejar solo la columna necesaria
	df13 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden13 = df13.sort_values('Hotels/Other lodgings', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden13.index:
		total = orden13['Hotels/Other lodgings'][value] + total
	media = total/len(orden13)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad13 = orden13['Hotels/Other lodgings'].value_counts()

	#Ordenar de menor a mayor
	cantidad13 = cantidad13.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf13 = pd.DataFrame(cantidad13)
	graf13_reset = graf13.reset_index()
	graf13_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf13_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf13_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'HOTELS/OTHER LODGINGS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()


def zumos():

	#Dejar solo la columna necesaria
	df14 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden14 = df14.sort_values('Juice bars', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden14.index:
		total = orden14['Juice bars'][value] + total
	media = total/len(orden14)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad14 = orden14['Juice bars'].value_counts()

	#Ordenar de menor a mayor
	cantidad14 = cantidad14.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf14 = pd.DataFrame(cantidad14)
	graf14_reset = graf14.reset_index()
	graf14_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf14_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf14_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'JUICE BARS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def galerias():

	#Dejar solo la columna necesaria
	df15 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden15 = df15.sort_values('Art galleries', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden15.index:
		total = orden15['Art galleries'][value] + total
	media = total/len(orden15)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad15 = orden15['Art galleries'].value_counts()

	#Ordenar de menor a mayor
	cantidad15 = cantidad15.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf15 = pd.DataFrame(cantidad15)
	graf15_reset = graf15.reset_index()
	graf15_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf15_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf15_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'ART GALLERIES REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def discos():

	#Dejar solo la columna necesaria
	df16 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden16 = df16.sort_values('Dance clubs', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden16.index:
		total = orden16['Dance clubs'][value] + total
	media = total/len(orden16)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad16 = orden16['Dance clubs'].value_counts()

	#Ordenar de menor a mayor
	cantidad16 = cantidad16.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf16 = pd.DataFrame(cantidad16)
	graf16_reset = graf16.reset_index()
	graf16_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf16_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf16_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'DANCE CLUBS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def piscinas():

	#Dejar solo la columna necesaria
	df17 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden17 = df17.sort_values('Swimming pools', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden17.index:
		total = orden17['Swimming pools'][value] + total
	media = total/len(orden17)
	print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad17 = orden17['Swimming pools'].value_counts()

	#Ordenar de menor a mayor
	cantidad17 = cantidad17.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf17 = pd.DataFrame(cantidad17)
	graf17_reset = graf17.reset_index()
	graf17_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf17_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf17_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'SWIMMING POOLS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def gimnasios():

	#Dejar solo la columna necesaria
	df18 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden18 = df18.sort_values('Gyms', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden18.index:
		total = orden18['Gyms'][value] + total
	media = total/len(orden18)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad18 = orden18['Gyms'].value_counts()

	#Ordenar de menor a mayor
	cantidad18 = cantidad18.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf18 = pd.DataFrame(cantidad18)
	graf18_reset = graf18.reset_index()
	graf18_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf18_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf18_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'GYMS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def panaderias():

	#Dejar solo la columna necesaria
	df19 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden19 = df19.sort_values('Bakeries', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden19.index:
		total = orden19['Bakeries'][value] + total
	media = total/len(orden19)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad19 = orden19['Bakeries'].value_counts()

	#Ordenar de menor a mayor
	cantidad19 = cantidad19.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf19 = pd.DataFrame(cantidad19)
	graf19_reset = graf19.reset_index()
	graf19_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf19_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf19_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'BAKERIES REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def spas():

	#Dejar solo la columna necesaria
	df20 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden20 = df20.sort_values('Beauty & Spas', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden20.index:
		total = orden20['Beauty & Spas'][value] + total
	media = total/len(orden20)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad20 = orden20['Beauty & Spas'].value_counts()

	#Ordenar de menor a mayor
	cantidad20 = cantidad20.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf20 = pd.DataFrame(cantidad20)
	graf20_reset = graf20.reset_index()
	graf20_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf20_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf20_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'BEAUTY & SPAS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def cafes():

	#Dejar solo la columna necesaria
	df21 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden21 = df21.sort_values('Cafes', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden21.index:
		total = orden21['Cafes'][value] + total
	media = total/len(orden21)
	print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad21 = orden21['Cafes'].value_counts()

	#Ordenar de menor a mayor
	cantidad21 = cantidad21.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf21 = pd.DataFrame(cantidad21)
	graf21_reset = graf21.reset_index()
	graf21_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf21_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf21_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'CAFES REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def miradores():

	#Dejar solo la columna necesaria
	df22 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden22 = df22.sort_values('View points', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden22.index:
		total = orden22['View points'][value] + total
	media = total/len(orden22)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad22 = orden22['View points'].value_counts()

	#Ordenar de menor a mayor
	cantidad22 = cantidad22.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf22 = pd.DataFrame(cantidad22)
	graf22_reset = graf22.reset_index()
	graf22_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf22_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf22_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'VIEW POINTS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def monumentos():

	#Dejar solo la columna necesaria
	df23 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24]], axis='columns')

	#Ordenar valores del 0 al 5
	orden23 = df23.sort_values('Monuments', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden23.index:
		total = orden23['Monuments'][value] + total
	media = total/len(orden23)
	#print("{:.2f}".format(media))

	#Cuantas veces sale cada nota
	cantidad23 = orden23['Monuments'].value_counts()

	#Ordenar de menor a mayor
	cantidad23 = cantidad23.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf23 = pd.DataFrame(cantidad23)
	graf23_reset = graf23.reset_index()
	graf23_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf23_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf23_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'MONUMENTS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))

	fig.show()

def jardines():

	#Dejar solo la columna necesaria
	df24 = ds.drop(ds.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]], axis='columns')

	#Ordenar valores del 0 al 5
	orden24 = df24.sort_values('Gardens', ascending=True)

	#Calcular nota media de las reviews
	total = 0
	for value in orden24.index:
		total = orden24['Gardens'][value] + total
	media = total/len(orden24)
	#print("{:.2f}".format(media))


	#Cuantas veces sale cada nota
	cantidad24 = orden24['Gardens'].value_counts()

	#Ordenar de menor a mayor
	cantidad24 = cantidad24.sort_index()

	#Crear un nuevo dataframe con valores y las veces que sale cada valor
	graf24 = pd.DataFrame(cantidad24)
	graf24_reset = graf24.reset_index()
	graf24_reset.columns = ['Values', 'Quantities']

	#Visualizar dataset final
	#print(graf24_reset)

	#Crear gráfico a partir de los datos
	fig = px.scatter(graf24_reset, x = 'Values', y = 'Quantities', color = 'Values', color_continuous_scale=px.colors.diverging.RdYlGn, title = 'GARDENS REVIEWS')
	fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGray')),
                  selector=dict(mode='markers'))
	fig.show()