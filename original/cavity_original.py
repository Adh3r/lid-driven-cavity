'''
import os
'''

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- CAVIDAD POR DIFERÉNCIAS FINITAS VECTORIZADO ---


# --- 1. CONFIGURACIÓN Y PARÁMETROS ---

# Dimensión de la malla
Nx = 101    
Ny = 101

# Geometria de la cavidad [m]
Lx = 1.0       
Ly = 1.0     

# Condiciones físicas
Re = 1000.0
U_lid = 1.0                 # Velocidad de la tapa [m/s]

# Propiedades del problema
nu = (U_lid * Lx) / Re      # Viscosidad cinemática [m^2/s]
h = Lx / (Nx - 1)           # Tamaño de celda [m]

# Discretización temporal
dt = 0.001                  # Paso de tiempo [s]
t_final = 100.0             # Tiempo máximo [s]

# Condiciones de estabilidad
check_interval = 50  
steady_tolerance = 1e-7
max_iter_poisson = 3000
tolerance_poisson = 1e-5 

# --- 2. INICIALIZACIÓN ---

# Inicialización fisica
psi = np.zeros((Ny, Nx))    # Función de corriente [m^2/s]
xi = np.zeros((Ny, Nx))  # Vorticidad [1/s]
u = np.zeros((Ny, Nx))      # Velocidad Horizontal [m/s]
v = np.zeros((Ny, Nx))      # Velocidad Vertical [m/s]

# Inicialización graficador 
x = np.linspace(0, Lx, Nx)  # Eje x
y = np.linspace(0, Ly, Ny)  # Eje y
X, Y = np.meshgrid(x, y)    # Malla
plot_interval = 200         # Actualizamos visualmente cada 10 pasos

'''
# Gestión de archivos
frames_dir = 'frames_temp'
os.makedirs(frames_dir, exist_ok=True)

base_prefix = f'Cavidad_Re{int(Re)}_{Nx}x{Ny}'
counter = 0
while True:
    final_base_name = f'{base_prefix}_{counter}' if counter > 0 else base_prefix
    if not os.path.exists(f'{final_base_name}.mp4'):
        break
    counter += 1
'''

# Historiales estadísticas
history_t = []
history_residual = []
history_cfl = []
history_ek = []
frame_files = []


# --- 3. CONFIGURACIÓN GRÁFICA ---
plt.ion()
fig, ax = plt.subplots(figsize=(7, 8.5))

# Estetica del gráfico
plt.subplots_adjust(top=0.82, bottom=0.15, left=0.16, right=0.84, hspace=0, wspace=0)
divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='4%', pad=0.6)
fixed_levels = np.linspace(-20, 20, 100)

# Arranque
print(f'Simulación iniciada "Lid Driven Cavity"')
start_time = time.time()

# Printer de datos
print(f'Condiciones de físicas: Cavidad de {Lx}x{Ly} m^2, U_lid={U_lid} [m/s], Re={Re}')
print(f'Condiciones de malla: Malla de {Nx}x{Ny}, h={h} m, dt={dt} [m/s]')
print(f'Convergencia alcanzada en: {steady_tolerance} [s^-1]')

# --- 4. BUCLE PRINCIPAL ---
n = 0
while True:
    t_sim = n * dt      # Tiempo dentro de la simulación
    if t_sim >= t_final:
        print('\nSe alcanzó el tiempo máximo sin convergencia total.')
        break

    xi_old = xi.copy() 

    # --- A. Física versión vectorizada ---
    # NOTA COMPUTACIONAL:
    # Se sustituyen los bucles 'for' por 'slicing' de NumPy.
    # En lugar de iterar celda a celda (lento en Python), operamos con matrices
    # completas.

    # 1. Condiciones de frontera para Vorticidad
    # Pared Superior (Top) - Usamos slicing [-1, :] para la última fila
    xi[-1, 1:-1] = -2.0 * (psi[-2, 1:-1] + h * U_lid) / (h**2)
    # Pared Inferior (Bottom)
    xi[0, 1:-1]  = -2.0 * psi[1, 1:-1] / (h**2)
    # Pared Izquierda (Left)
    xi[1:-1, 0]  = -2.0 * psi[1:-1, 1] / (h**2)
    # Pared Derecha (Right)
    xi[1:-1, -1] = -2.0 * psi[1:-1, -2] / (h**2)
    # NOTA TEÓRICA:
    # Se mantiene la expansión de Taylor para conectar vorticidad en pared con
    # la función de corriente interna, imponiendo velocidad cero (no-deslizamiento).

    # 2. Transporte de Vorticidad (Navier-Stokes Interior)
    # Definimos "vistas" de los vecinos para operar en bloque.
    # [1:-1, 1:-1] representa todos los puntos interiores (sin paredes)
    w_c = xi[1:-1, 1:-1] # Centro
    w_l = xi[1:-1, 0:-2] # Izquierda (i-1)
    w_r = xi[1:-1, 2:]   # Derecha   (i+1)
    w_b = xi[0:-2, 1:-1] # Abajo     (j-1)
    w_t = xi[2:, 1:-1]   # Arriba    (j+1)
    
    u_c = u[1:-1, 1:-1]
    v_c = v[1:-1, 1:-1]

    # Laplaciano de la difusión (Diferencias centradas vectorizadas)
    laplacian = (w_r - 2*w_c + w_l)/h**2 + (w_t - 2*w_c + w_b)/h**2
    
    # Advección (Upwind explícito con max/min)
    # Si u > 0, el fluido viene de la izquierda (w_c - w_l)
    adv_x = (np.maximum(u_c, 0) * (w_c - w_l) + np.minimum(u_c, 0) * (w_r - w_c)) / h
    adv_y = (np.maximum(v_c, 0) * (w_c - w_b) + np.minimum(v_c, 0) * (w_t - w_c)) / h
    # NOTA TEÓRICA:
            # Se aplica el esquema diferencias adelante según la dirección del flujo,
            # positiva o negativa ya que el uso de diferencias centradas para la 
            # advección es numéricamente inestable. Al tomar la información desde
            # de donde viene el fluido, la solución se estabiliza debido a la 
            # introducción una ligera difusión numérica.

    # Actualización de la vorticidad
    xi[1:-1, 1:-1] = w_c + dt * (nu * laplacian - adv_x - adv_y)

    # 3. Función de Corriente
    # Usamos metodo de relajación = (diferencias centradas de segundo orden para la eq de la vorticidad)
    # Poisson
    for _ in range(max_iter_poisson):
        psi_new = psi.copy()
        psi_new[1:-1, 1:-1] = 0.25 * (psi[1:-1, 2:] + psi[1:-1, :-2] +
                                      psi[2:, 1:-1] + psi[:-2, 1:-1] +
                                      h**2 * xi[1:-1, 1:-1])
        if _ % 50 == 0:
            # Comparamos la psi actual con la que guardamos al inicio del bucle
            if np.max(np.abs(psi_new - psi)) < tolerance_poisson:
                psi = psi_new
                break
        psi = psi_new
    # NOTA TEÓRICA:
    # Se impone un límite de iteraciones ya que, aún que aspiramos a que la 
    # solución de psi cumpla el margen de 'tolerance_poisson' dependiendo de las
    # condiciones iniciales no vale la pena computacionalmente gastar tanto tiempo
    # para incrementar la presición dada la naturaleza del proyecto.

    # 4. Velocidades (Diferencias centradas)
    u[1:-1, 1:-1] =  (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)

    # Corrección tapa para imponer en todo momento la condicion inicial U_lid
    u[-1, :] = U_lid

    # --- B. Comprobación de Estabilidad ---
    if n % check_interval == 0 and n > 0:
        # Calculamos cuánto ha cambiado la vorticidad respecto al paso anterior
        change = np.max(np.abs(xi - xi_old))
        
        if change < steady_tolerance:
            print(f'\nESTABILIDAD ALCANZADA en el paso {n} (t={t_sim:.3f}s)')
            print(f'Variación final: {change:.2e} < {steady_tolerance}')
            break

    # --- C. Datos en consola ---
    if n % 100 == 0: # Calculamos cada 10 pasos para no saturar CPU
        # Calculamos energía cinética
        ek = 0.5 * np.sum(u**2 + v**2) * h**2
        # Cuanto más cercano a zero más presición de la solución
        current_res = np.max(np.abs(xi - xi_old))
        # Número de Courant (Monitoreo del código)(siempre que esté por debajo de 1 todo es estable)
        cfl = (np.max(np.abs(u)) + np.max(np.abs(v))) * dt / h

        history_t.append(t_sim)
        history_residual.append(current_res)
        history_ek.append(ek)
        history_cfl.append(cfl)

        print(f'Paso {n} | t={t_sim:.2f} [s]| Ek={ek:.4f} [J]| Var={current_res:.1e} [s^-1]| CFL={cfl:.3f}')

    # --- D. Visualización ---
    if n % plot_interval == 0:
        ax.clear()
        cax.clear()
        
        xi_plot = np.clip(xi, -15, 15)
        # NOTA TEÓRICA:
        # Se usa np.clip para filtrar los valores de las esquinas superiores
        # donde se presentan singularidades por las condiciones de contorno:
        # 0 en las paredes verticales y 1 en la tapa. Esta discontinuidad de
        # velocidad genera teóricamente vorticidad infinita en el cálculo de
        # tal manera que si lo incluyeramos en la escala de colores de la 
        # graficación no se veria claramente lo que queremos representar.
        cf = ax.contourf(X, Y, xi_plot, levels=fixed_levels, cmap='jet')
        
        ax.set_title(f'Simulación (Re={Re}) | $U_{{lid}}$={U_lid} m/s s\nt={t_sim:.3f}', fontsize=11)
        ax.set_xlabel('Pared X [m]', fontsize=10)
        ax.set_ylabel('Pared Y [m]', fontsize=10)
        ax.set_xticks([0, Lx])
        ax.set_yticks([0, Ly])
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
        cbar.set_label(r'Vorticidad $\xi$ [s$^{-1}$]', fontsize=10)
        cbar.set_ticks([-20, -10, 0, 10, 20])
        cbar.ax.tick_params(labelsize=9)
        
        plt.pause(0.0001)

        '''
        frame_name = os.path.join(frames_dir, f'frame_{n:06d}.png')
        fig.savefig(frame_name, dpi=100)
        frame_files.append(frame_name)
        '''
    
    n += 1

print(f'Simulación finalizada en {time.time() - start_time:.2f} s')
plt.ioff()
plt.show()

'''
# Generador de video
print('Generando video MP4...')
fps = 30
video_name = f'{final_base_name}.mp4'

try:
    if not frame_files:
        raise Exception('No se han generado frames.')

    img0 = plt.imread(frame_files[0])
    height, width, layers = img0.shape
    
    dpi_video = 100
    fig_w = width / dpi_video
    fig_h = height / dpi_video
    
    fig_video = plt.figure(figsize=(fig_w, fig_h), dpi=dpi_video)
    
    ax_video = fig_video.add_axes([0, 0, 1, 1])
    ax_video.set_axis_off()

    writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
    
    with writer.saving(fig_video, video_name, dpi=dpi_video):
        for i, f in enumerate(frame_files):
            if i % 10 == 0: print(f'Procesando frame {i}/{len(frame_files)}...', end='\r')
            
            img = plt.imread(f)
            
            ax_video.clear()
            ax_video.set_axis_off()
            ax_video.imshow(img)
            
            writer.grab_frame()

    plt.close(fig_video)
    print(f'\nVideo guardado: {video_name}')

except Exception as e:
    print(f'Error generando video: {e}')
'''

# --- 5. GRAFICADOR DE DATOS EXTRA ---

# 1. Velocidades
fig_vel, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Perfil de U a lo largo de la línea vertical central
y_center_line = np.linspace(0, Ly, Ny)
u_center_line = u[:, Nx//2]  # Columna central

ax1.plot(u_center_line, y_center_line, 'b-o', markersize=3, label=f'Re={Re}')
ax1.set_title('Perfil U en la línea vertical central')
ax1.set_xlabel('Velocidad Horizontal u [m/s]')
ax1.set_ylabel('Posición Y [m]')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Perfil de V a lo largo de la línea horizontal central
x_center_line = np.linspace(0, Lx, Nx)
v_center_line = v[Ny//2, :]  # Fila central

ax2.plot(x_center_line, v_center_line, 'r-o', markersize=3, label=f'Re={Re}')
ax2.set_title('Perfil V en la línea horizontal central')
ax2.set_xlabel('Posición X [m]')
ax2.set_ylabel('Velocidad Vertical v [m/s]')
ax2.grid(True, alpha=0.3)
ax2.legend()

'''
fig_vel.tight_layout()
fig_vel.savefig(f'{final_base_name}_Perfiles_de_velocidad.png')
'''

# 2. Convergencia
fig_stats, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

# Energia cinética
ax1.plot(history_t, history_ek, 'g-', lw=2, label='Energía Cinética')
ax1.set_ylabel('Energía Cinética Total $E_k$ [J]', color='g', fontsize=10)
ax1.set_title(f'Historial de Convergencia (Re={Re})', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower right')

# Residual
ax2.semilogy(history_t, history_residual, 'm-', lw=1.5)
ax2.set_ylabel(r'Residuo (max $|\Delta\xi|$) [s$^{-1}$]', color='m', fontsize=10)
ax2.set_xlabel('Tiempo de Simulación [s]', fontsize=10)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend(loc='upper right')

'''
fig_stats.tight_layout()
fig_stats.savefig(f'{final_base_name}_Convergencia.png')
'''

plt.show()