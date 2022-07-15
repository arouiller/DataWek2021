import sys
import click
from numpy import str0

@click.command()
@click.option('--name', default='', type=str, help='Tu nombre')
@click.option('--repeticiones', default=1, type=int, help='Cantidad de repeticiones')


def run(name, repeticiones):
   print("")
   print("*****************************************************")
   for i in range(repeticiones):
      print('Hola', name, ' esta es la repeticion ', str(i+1))
   print("*****************************************************")
   print("")

if __name__ == "__main__":
    run()