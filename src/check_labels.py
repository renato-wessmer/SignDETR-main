import os
import json

def check_actual_label_mapping():
    """Verifica qual mapeamento está sendo usado nos arquivos de label"""
    
    # Caminho para os labels de treino
    labels_path = 'data/train/labels/'
    
    # Verificar se a pasta existe
    if not os.path.exists(labels_path):
        print(f"❌ Pasta não encontrada: {labels_path}")
        return
    
    # Pegar os primeiros 10 arquivos .txt
    txt_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')][:10]
    
    if not txt_files:
        print("❌ Nenhum arquivo .txt encontrado!")
        return
    
    print("=== VERIFICANDO MAPEAMENTO REAL NOS LABELS ===")
    print(f"Encontrados {len(txt_files)} arquivos para análise")
    
    class_counts = {}
    
    for txt_file in txt_files:
        file_path = os.path.join(labels_path, txt_file)
        print(f"\n📄 {txt_file}:")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                print("  (arquivo vazio)")
                continue
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    
                    # Contar ocorrências de cada classe
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    
                    print(f"  ID {class_id} → bbox {bbox}")
                else:
                    print(f"  ⚠️ Formato inválido: {line.strip()}")
                    
        except Exception as e:
            print(f"  ❌ Erro ao ler arquivo: {e}")
    
    # Resumo das classes encontradas
    print(f"\n📊 RESUMO DOS IDs ENCONTRADOS:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  ID {class_id}: {count} ocorrências")

# Executar a verificação
if __name__ == "__main__":
    check_actual_label_mapping()