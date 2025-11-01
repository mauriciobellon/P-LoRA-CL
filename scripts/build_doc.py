#!/usr/bin/env python3
"""
Script para compilar todos os arquivos markdown dos capítulos em um único documento.
Mantém a hierarquia completa do trabalho usando títulos markdown.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional


def extract_number_from_path(file_path: str) -> Tuple[int, ...]:
    """
    Extrai números do caminho do arquivo para ordenação.
    Exemplo: "1. Introdução/1.1. Contextualização/1.1.1. Arquivo.md" -> (1, 1, 1)
    """
    numbers = []
    # Extrai todos os números do caminho
    matches = re.findall(r'(\d+)\.', file_path)
    for match in matches:
        numbers.append(int(match))
    return tuple(numbers)


def extract_section_from_path(file_path: str) -> Optional[str]:
    """
    Extrai o número da seção (1.1, 1.2, etc) do caminho do arquivo.
    Retorna None se não encontrar seção intermediária.
    """
    # Procura por padrão tipo "1.1. Nome" ou "1.2. Nome" nas pastas
    match = re.search(r'/(\d+\.\d+)\.\s+([^/]+)/', str(file_path))
    if match:
        return match.group(1) + '. ' + match.group(2)
    return None


def get_section_name_from_path(file_path: str) -> Optional[str]:
    """
    Extrai o nome completo da seção (pasta) do caminho.
    Exemplo: "1. Introdução/1.1. Contextualização e motivação/1.1.1. Arquivo.md"
    -> "1.1. Contextualização e motivação"
    """
    parts = Path(file_path).parts
    # Procura por uma pasta que começa com número.número
    for part in parts:
        if re.match(r'^\d+\.\d+\.', part):
            return part
    return None


def get_file_title_level(file_path: str) -> int:
    """
    Determina o nível do título baseado na profundidade do caminho.
    Capítulos principais (1, 2, 3...) = nível 2 (##)
    Seções (pastas 1.1, 2.1...) = nível 3 (###)
    Subseções (arquivos 1.1.1, 2.1.1...) = nível 4 (####)
    Arquivos sem subseção (1.6) = nível 3 (###)
    """
    # Conta quantos níveis de numeração existem no caminho do arquivo
    # Verifica se é um arquivo (termina com .md) ou uma pasta
    file_path_str = str(file_path)

    # Se contém três números separados por ponto (ex: 1.1.1), é subseção = nível 4
    if re.search(r'\d+\.\d+\.\d+\.', file_path_str):
        return 4

    # Se contém apenas um número (ex: 1.6.md), é nível 3
    if re.match(r'^.*/\d+\.', file_path_str) and not re.search(r'\d+\.\d+\.', file_path_str):
        return 3

    # Por padrão, arquivos em subpastas são nível 4
    return 4


def get_section_title(file_path: str, content: str) -> str:
    """
    Extrai ou gera o título da seção baseado no caminho e conteúdo.
    """
    # Tenta extrair o título da primeira linha do conteúdo
    lines = content.strip().split('\n')
    if lines and lines[0].startswith('#'):
        # Remove o número prefixado do título se existir
        title = lines[0].lstrip('#').strip()
        # Remove números do início (ex: "1.1.1. " -> "")
        title = re.sub(r'^\d+\.\d*\.?\d*\.?\s*', '', title)
        return title

    # Se não encontrar título no conteúdo, gera do nome do arquivo
    filename = Path(file_path).stem
    # Remove números do início
    title = re.sub(r'^\d+\.\d*\.?\d*\.?\s*', '', filename)
    # Remove extensão .md se ainda estiver presente
    title = title.replace('.md', '')
    return title


def normalize_title_level(content: str, base_level: int) -> str:
    """
    Normaliza os níveis de título no conteúdo baseado na hierarquia.
    base_level é o nível do título principal da seção.
    """
    lines = content.split('\n')
    normalized_lines = []

    for line in lines:
        if line.strip().startswith('#'):
            # Conta quantos # existem no título original
            header_level = len(line) - len(line.lstrip('#'))
            # Se o conteúdo tem título com #, assume que é relativo ao nível base
            # e ajusta adequadamente
            if header_level == 1:
                # Se o conteúdo tem nível 1, mantém no nível base
                new_level = base_level
            else:
                # Outros níveis são incrementados do base
                new_level = base_level + (header_level - 1)

            if new_level < 1:
                new_level = 1
            if new_level > 6:
                new_level = 6

            # Reconstrói a linha com o nível correto
            title_text = line.lstrip('#').strip()
            # Remove números do início do título também
            title_text = re.sub(r'^\d+\.\d*\.?\d*\.?\s*', '', title_text)
            normalized_lines.append('#' * new_level + ' ' + title_text)
        else:
            normalized_lines.append(line)

    return '\n'.join(normalized_lines)


def compile_documentation(docs_dir: Path, output_file: Path) -> None:
    """
    Compila todos os arquivos markdown da pasta docs em um único documento.
    """
    # Encontra todos os arquivos .md (incluindo os da raiz e das subpastas)
    markdown_files = sorted(
        docs_dir.rglob('*.md'),
        key=lambda p: extract_number_from_path(str(p.relative_to(docs_dir)))
    )

    if not markdown_files:
        print(f"Erro: Nenhum arquivo markdown encontrado em {docs_dir}")
        return

    print(f"Encontrados {len(markdown_files)} arquivos markdown")

    compiled_content = []

    # Adiciona título principal
    compiled_content.append('# Rede Progressiva com LoRA Ortogonal para Aprendizado Contínuo')
    compiled_content.append('')

    last_chapter = None
    last_section = None

    for file_path in markdown_files:
        relative_path = file_path.relative_to(docs_dir)

        # Se o arquivo está na raiz de docs (ex: 0. Resumo.md, 6. Referências...)
        if len(relative_path.parts) == 1:
            # Arquivo na raiz: nível 2
            filename = Path(file_path).stem
            chapter_match = re.match(r'^(\d+)\.', filename)

            if chapter_match:
                current_chapter = chapter_match.group(1)

                # Adiciona título se mudou
                if current_chapter != last_chapter:
                    if last_chapter is not None:
                        # Remove todas as linhas vazias no final
                        while compiled_content and compiled_content[-1].strip() == '':
                            compiled_content.pop()
                        # Sempre adiciona uma linha vazia antes do ---
                        compiled_content.append('')
                        compiled_content.append('---')
                        compiled_content.append('')
                    compiled_content.append(f'## {filename}')
                    compiled_content.append('')
                    last_chapter = current_chapter
                    last_section = None

            print(f"Processando: {relative_path}")

            # Lê o conteúdo
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except Exception as e:
                print(f"Erro ao ler {file_path}: {e}")
                continue

            if not content:
                continue

            # Para arquivos na raiz, remove apenas o primeiro título se for igual ao nome do arquivo
            lines = content.split('\n')
            if lines and lines[0].startswith('#'):
                first_title = lines[0].lstrip('#').strip()
                # Remove números do início do título para comparação
                first_title_clean = re.sub(r'^\d+\.\d*\.?\d*\.?\s*', '', first_title)
                filename_clean = re.sub(r'^\d+\.\d*\.?\d*\.?\s*', '', filename)
                # Se o título do arquivo corresponde ao primeiro título do conteúdo, remove
                if first_title_clean.strip() == filename_clean.strip():
                    lines = lines[1:]
                # Normaliza os níveis de título restantes (aumenta em 1 nível já que o arquivo é nível 2)
                normalized_lines = []
                for line in lines:
                    if line.strip().startswith('#'):
                        header_level = len(line) - len(line.lstrip('#'))
                        new_level = min(header_level + 1, 6)  # Máximo nível 6
                        title_text = line.lstrip('#').strip()
                        normalized_lines.append('#' * new_level + ' ' + title_text)
                    else:
                        normalized_lines.append(line)
                lines = normalized_lines

            content_final = '\n'.join(lines).strip()
            if content_final:
                compiled_content.append(content_final)
                # Mantém linha vazia no final para separar do próximo capítulo

            continue

        # Arquivo em subpasta (processamento normal)
        # Extrai o capítulo principal
        chapter_match = re.search(r'^(\d+)\.', str(relative_path))
        current_chapter = chapter_match.group(1) if chapter_match else None

        # Adiciona título do capítulo se mudou
        if current_chapter and current_chapter != last_chapter:
            if last_chapter is not None:
                # Remove todas as linhas vazias no final
                while compiled_content and compiled_content[-1].strip() == '':
                    compiled_content.pop()
                # Sempre adiciona uma linha vazia antes do ---
                compiled_content.append('')
                compiled_content.append('---')
                compiled_content.append('')
            # Extrai nome do capítulo do caminho (mantém número e nome)
            chapter_name = relative_path.parts[0]
            compiled_content.append(f'## {chapter_name}')
            compiled_content.append('')
            last_chapter = current_chapter
            last_section = None  # Reset seção quando muda capítulo

        # Extrai a seção (pasta intermediária) se existir
        # Só adiciona título de seção se o arquivo está dentro de uma subpasta
        # (não para arquivos diretos como 1.6.md)
        path_parts = relative_path.parts
        current_section = None

        # Se o arquivo está em uma subpasta (tem mais de 1 parte após o capítulo)
        if len(path_parts) > 2:
            # A segunda parte é a seção (pasta intermediária)
            current_section = path_parts[1]
        elif len(path_parts) == 2:
            # Arquivo direto no capítulo (como 1.6.md), não tem seção intermediária
            current_section = None

        # Adiciona título da seção se mudou e existe
        if current_section and current_section != last_section:
            compiled_content.append(f'### {current_section}')
            compiled_content.append('')
            last_section = current_section

        print(f"Processando: {relative_path}")

        # Lê o conteúdo do arquivo
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            print(f"Erro ao ler {file_path}: {e}")
            continue

        if not content:
            continue

        # Determina o nível do título baseado na estrutura do caminho
        # Arquivos diretos no capítulo (sem subpasta) = nível 3
        # Arquivos em subpastas = nível 4
        path_parts = relative_path.parts
        if len(path_parts) == 2:
            # Arquivo direto no capítulo (ex: 1.6.md)
            level = 3
        else:
            # Arquivo em subpasta (ex: 1.1.1.md)
            level = 4

        # Extrai o título da seção
        section_title = get_section_title(str(relative_path), content)

        # Para arquivos em subpastas (subseções), inclui o número completo no título
        if len(path_parts) == 3:
            filename = Path(file_path).stem
            # Extrai o número completo do arquivo (ex: "1.1.1. Desafio..." -> "1.1.1")
            number_match = re.match(r'^(\d+\.\d+\.\d+)\.', filename)
            if number_match:
                section_title = f"{number_match.group(1)}. {section_title}"

        # Para arquivos diretos no capítulo, inclui o número no título
        elif len(path_parts) == 2:
            filename = Path(file_path).stem
            # Extrai o número completo do arquivo (ex: "1.6. Estrutura do trabalho" -> "1.6")
            number_match = re.match(r'^(\d+(?:\.\d+)?)\.', filename)
            if number_match:
                section_title = f"{number_match.group(1)}. {section_title}"

        # Adiciona o título da seção como cabeçalho
        compiled_content.append('#' * level + ' ' + section_title)
        compiled_content.append('')

        # Normaliza os níveis de título no conteúdo
        normalized_content = normalize_title_level(content, level)

        # Remove o título original se existir (primeira linha)
        lines = normalized_content.split('\n')
        if lines and lines[0].startswith('#'):
            lines = lines[1:]

        # Adiciona o conteúdo (sem o título original)
        content_without_title = '\n'.join(lines).strip()
        if content_without_title:
            compiled_content.append(content_without_title)
            compiled_content.append('')

    # Escreve o arquivo compilado
    final_content = '\n'.join(compiled_content)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print(f"\nDocumento compilado com sucesso: {output_file}")
        print(f"Total de arquivos processados: {len(markdown_files)}")
    except Exception as e:
        print(f"Erro ao escrever arquivo: {e}")
        return


def main():
    """Função principal."""
    # Define caminhos
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / 'docs/paper'
    output_file = project_root / "docs/paper.md"

    # Verifica se o diretório existe
    if not docs_dir.exists():
        print(f"Erro: Diretório não encontrado: {docs_dir}")
        return

    print(f"Diretório de origem: {docs_dir}")
    print(f"Arquivo de saída: {output_file}")
    print()

    # Compila a documentação
    compile_documentation(docs_dir, output_file)


if __name__ == '__main__':
    main()
