import os
import glob

replacements = {
    'text-gray-300': 'text-[#4A261D]',
    'text-gray-200': 'text-[#2F1A13]',
    'text-gray-400': 'text-[#663A2C]',
    'text-green-300': 'text-[#381E15]',
    'text-green-400': 'text-[#381E15]',
    'bg-[#001a00]/60': 'bg-white/40',
    'bg-[#001a00]/70': 'bg-white/50',
    'bg-[#002b1a]/50': 'bg-white/40',
    'bg-[#002b1a]/60': 'bg-white/50',
    'bg-[#0a0a0a]/50': 'bg-white/40',
    'border-[#00ff88]/20': 'border-[#A65B45]/40',
    'border-[#00ff88]/30': 'border-[#A65B45]/40',
    'border-green-400/30': 'border-[#A65B45]/40',
    'border-cyan-400/30': 'border-[#A65B45]/40',
    'from-green-400': 'from-[#381E15]',
    'to-cyan-300': 'to-[#4A261D]',
    'bg-gradient-to-r from-[#381E15] to-[#4A261D]': 'bg-[#E5E9C9]', 
    'bg-gradient-to-r from-green-400 to-cyan-400': 'bg-[#E5E9C9]',
    'bg-gradient-to-r from-green-600 to-emerald-500': 'bg-[#E5E9C9]',
    'shadow-[0_0_25px_#00ff8850]': 'shadow-lg',
    'shadow-[0_0_20px_#00ff8880]': 'shadow-md',
    'shadow-[0_0_20px_#00ff8850]': 'shadow-md',
    'text-cyan-400': 'text-[#2F1A13]',
    'text-white': 'text-[#2F1A13]',
    'text-gray-100': 'text-[#2F1A13]',
    'placeholder-gray-400': 'placeholder-[#663A2C]',
    'bg-black/40': 'bg-transparent',
    'border-white/10': 'border-[#A65B45]/20',
    'text-gray-500': 'text-[#663A2C]',
    'text-green-700': 'text-[#4A261D]',
    'text-green-100': 'text-[#2F1A13]',
    'text-green-200/80': 'text-[#4A261D]',
    'border-green-800': 'border-[#A65B45]/40',
    'bg-green-900/20': 'bg-white/40',
    'bg-black': 'bg-transparent',
}

files = glob.glob('templates/*.html')
for f in files:
    # skip index.html and base.html since we just wrote them explicitly and they are already themed!
    if f.endswith('index.html') or f.endswith('base.html'):
        continue
    with open(f, 'r') as file:
        content = file.read()
    for k, v in replacements.items():
        content = content.replace(k, v)
    with open(f, 'w') as file:
        file.write(content)
print("done replacing")
