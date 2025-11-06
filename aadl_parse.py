import re

class AADLParser:
    def __init__(self, aadl_code: str):
        self.aadl_code = aadl_code
        self.components = None

    def remove_aadl_comments(self) -> str:
        """
        移除注释
        """
        cleaned_lines = []
        for line in self.aadl_code.splitlines():
            idx = line.find("--")
            if idx != -1:
                line = line[:idx]
            line = line.rstrip()
            if line.strip():
                cleaned_lines.append(line)
        self.aadl_code = "\n".join(cleaned_lines)
        return "\n".join(cleaned_lines)

    def extract_aadl_base_info(self) -> dict:
        """
        检查public
        """
        # print(self.aadl_code)
        package_pattern = re.compile(
            r'package\s+([A-Za-z0-9_]+)\b(.*?)end\s+\1\s*;',
            re.DOTALL | re.IGNORECASE
        )

        match = package_pattern.search(self.aadl_code)
        if not match:
            raise ValueError("未找到合法的 package 块")

        package_name = match.group(1)
        package_content = match.group(2).strip()

        public_split = re.split(r'\bpublic\b', package_content, flags=re.IGNORECASE)
        if len(public_split) < 2:
            print(f"package {package_name} 缺少public段, 已自动插入")

            fixed_content = "public\n" + package_content

            full_old_block = match.group(0)
            full_new_block = f"package {package_name}\n{fixed_content}\nend {package_name};"
            self.aadl_code = self.aadl_code.replace(full_old_block, full_new_block)

            public_split = re.split(r'\bpublic\b', fixed_content, flags=re.IGNORECASE)
        else:
            fixed_content = package_content

        after_public = public_split[1].strip()
        with_clauses = re.findall(r'with\s+([A-Za-z0-9_:]+)\s*;', after_public)

        """
        return {
            'package_name': package_name,
            'with_clauses': with_clauses,
            'body': after_public,
            'fixed_code': self.aadl_code
        }
        """
        return self.aadl_code
        
    def split_components(self, package_body: str) -> list:
        """
        切割组件
        """
        components = []
        keywords = r'system|process|thread|subprogram|device|data|abstract|bus|memory|processor'
        component_pattern = re.compile(
            rf'\b(?P<kind>{keywords})\s+implementation\s+([A-Za-z0-9_.]+)\b|'   # 正常 implementation
            rf'\b(?P<kind2>{keywords})\s+([A-Za-z0-9_.]+)\b|'                   # 正常 type
            rf'\bimplementation\s+([A-Za-z0-9_.]+)\b',     # 缺少关键词的 implementation
            re.IGNORECASE
        )

        pos = 0
        length = len(package_body)

        while pos < length:
            match = component_pattern.search(package_body, pos)
            if not match:
                break

            line_end = package_body.find('\n', match.start())
            line_end = length if line_end == -1 else line_end
            line = package_body[match.start():line_end].strip()
            if line == 'data port;':
                pos = match.end()
                continue

            if match.group('kind'):
                kind = match.group('kind').lower() + ' implementation'
                name = match.group(2)
            elif match.group('kind2'):
                kind = match.group('kind2').lower()
                name = match.group(4)
            else:
                name = match.group(5)
                real_name = name.split(".impl")[0].split(".IMPL")[0]
                real_type = None
                for n in components:
                    if n.get("name") == real_name:
                        real_type = n.get("type")
                        break

                if real_type:
                    print(f"implementation {name} 缺少关键词, 推断为 {real_type}")
                    package_body = (
                        package_body[:match.start()] +
                        f"{real_type} implementation {name}" +
                        package_body[match.end():]
                    )
                    match = component_pattern.search(package_body, match.start())
                    kind = real_type + " implementation"
                else:
                    print(f"implementation {name} 缺少关键词, 但未找到对应type, 暂不修复")
                    kind = "unknown implementation"

            start_idx = match.start()

            end_any_pattern = re.compile(r'end\b\s*([A-Za-z0-9_.]+)?\s*;', re.IGNORECASE)
            end_any_match = end_any_pattern.search(package_body, match.end())

            next_match = component_pattern.search(package_body, match.end())

            if end_any_match is None or (end_any_match.group(1) is None) or (end_any_match.group(1) != name):
                content_end = next_match.start() if next_match else length
                content = package_body[match.end():content_end].strip()
                fixed_end_line = f"end {name};\n"

                if next_match:
                    raw_text = package_body[start_idx:content_end] + "\n" + fixed_end_line
                else:
                    raw_text = package_body[start_idx:content_end] + "\n" + fixed_end_line

                if end_any_match is None:
                    error = '缺少end'
                    print(f"组件 {kind} {name} 缺少end, 已自动补全")
                elif end_any_match.group(1) is None:
                    error = 'end缺少名称'
                    print(f"组件 {kind} {name} end缺少名称, 已自动补全")
                else:
                    error = f"名称不一致: 开头 {name}, 结尾 {end_any_match.group(1)}"
                    print(f"组件 {kind} {name} {error}, 已修正")

                components.append({
                    'type': kind,
                    'name': name,
                    'content': content,
                    'raw_text': raw_text,
                    'error': error,
                    'sub_block': []
                })

                pos = content_end
            else:
                content = package_body[match.end():end_any_match.start()].strip()
                raw_text = package_body[start_idx:end_any_match.end()].strip()
                components.append({
                    'type': kind,
                    'name': name,
                    'content': content,
                    'raw_text': raw_text,
                    'error': None,
                    'sub_block': []
                })
                pos = end_any_match.end()
        self.components = components
        return components
    
    def fix_implementation_consistency(self, components):
        """
        1. 名称未以.impl / .IMPL结尾，则补上.impl
        2. 没有对应的type组件，则自动生成一个
        """
        new_components = []
        existing_names = {c['name'].lower(): c for c in components if "implementation" not in c['type']}
        added_types = []

        for comp in components:
            kind = comp['type']
            name = comp['name']
            error_log = comp.get('error', None)

            if "implementation" in kind.lower():
                base_kind = kind.lower().replace(" implementation", "").strip()

                if not name.lower().endswith(".impl"):
                    fixed_name = name + ".impl"
                    print(f"组件 {kind} 名称 '{name}' 缺少.impl后缀, 已更名为 '{fixed_name}'")
                    if base_kind == name:
                        comp['raw_text'] = re.sub(
                            rf'\b{name}\b',
                            fixed_name,
                            comp['raw_text'],
                            count=2
                        )
                        comp['raw_text'] = re.sub(
                            rf'\b{fixed_name}\b',
                            name,
                            comp['raw_text'],
                            count=1
                        )

                    else:
                        comp['raw_text'] = re.sub(
                            rf'\b{name}\b',
                            fixed_name,
                            comp['raw_text'],
                            count=1
                        )
                    comp['raw_text'] = re.sub(
                        rf'end\s+{re.escape(name)}\s*;',
                        f'end {fixed_name};',
                        comp['raw_text']
                    )
                    comp['content'] = comp['content'].replace(name, fixed_name)
                    if base_kind == name:
                        comp['content'] = comp['content'].replace(fixed_name, name, 1)
                    comp['name'] = fixed_name
                    name = fixed_name
                    comp['error'] = (error_log or '') + " | 缺少.impl后缀, 已修复"

                base_name = name.split(".impl")[0].split(".IMPL")[0]
                if base_name.lower() not in existing_names:
                    print(f"组件 {kind} '{name}' 缺少对应的type '{base_name}'，已补全")

                    new_type = {
                        'type': base_kind,
                        'name': base_name,
                        'content': '',
                        'raw_text': f"{base_kind} {base_name}\nend {base_name};",
                        'error': '缺少对应type, 已补全',
                        'sub_block': []
                    }

                    added_types.append(new_type)
                    existing_names[base_name.lower()] = new_type

            new_components.append(comp)

        new_components.extend(added_types)
        self.aadl_code = new_components
        return new_components

    def fix_keyword_name_conflicts_name_only(self, components):
        """
        修复组件名称与AADL关键词冲突
        """
        aadl_keywords = [
            'system', 'process', 'thread', 'subprogram', 'device', 'data', 'abstract', 'bus', 'memory', 'processor'
        ]
        counter = 0

        for comp in components:
            old_name = comp['name']
            kind = comp.get('type', '')
            error_log = comp.get('error', '')

            if old_name.lower() in aadl_keywords:
                counter += 1
                new_name = f"{old_name}_{counter}"

                raw_text_fixed = re.sub(
                    rf'(\b{re.escape(old_name)}\b\s+){re.escape(old_name)}',
                    rf'\1' + new_name,
                    comp['raw_text']
                )
                raw_text_fixed = re.sub(
                    rf'end\s+{re.escape(old_name)}\s*;',
                    f'end {new_name};',
                    raw_text_fixed
                )

                content_fixed = comp['content'].replace(old_name, new_name)

                comp['name'] = new_name
                comp['raw_text'] = raw_text_fixed
                comp['content'] = content_fixed

                msg = f"组件名称 '{old_name}' 为关键词，已修改为 '{new_name}'"
                comp['error'] = (error_log + "----||----\n" if error_log else "") + msg

                print(f"组件 {kind} 名称 '{old_name}' -> '{new_name}'")

                base_name = old_name
                for other in components:
                    if other is comp:
                        continue
                    if "implementation" in other.get('type', '').lower():
                        impl_base = other['name'].split(".impl")[0].split(".IMPL")[0]
                        if impl_base == base_name:
                            impl_suffix = ".impl" if other['name'].lower().endswith(".impl") else ""
                            impl_new_name = new_name + impl_suffix

                            prefix_match = re.match(r'^(\w+\s+implementation\s+)', other['raw_text'])
                            prefix = prefix_match.group(1) if prefix_match else ''
                            other['raw_text'] = re.sub(
                                rf'(\b{re.escape(prefix)}){re.escape(other["name"])}',
                                rf'\1' + impl_new_name,
                                other['raw_text']
                            )

                            other['raw_text'] = re.sub(
                                rf'end\s+{re.escape(other["name"])}\s*;',
                                f'end {impl_new_name};',
                                other['raw_text']
                            )
                            other['content'] = other['content'].replace(other['name'], impl_new_name)
                            other['name'] = impl_new_name
                            other_msg = f"对应implementation名称也修改为 '{impl_new_name}'"
                            other['error'] = (other.get('error', '') + "----||----\n" if other.get('error') else "") + other_msg
                            print(f"对应implementation组件名称 -> '{impl_new_name}'")
                    else:
                        if other['name'] == base_name:
                            prefix_match = re.match(r'(\b\w+\b\s+)', other['raw_text'])
                            prefix = prefix_match.group(1) if prefix_match else ''
                            other['raw_text'] = re.sub(
                                rf'(\b{re.escape(prefix)}){re.escape(other["name"])}',
                                rf'\1' + new_name,
                                other['raw_text']
                            )
                            other['content'] = other['content'].replace(other['name'], new_name)
                            other['name'] = new_name
                            other_msg = f"对应type名称也修改为 '{new_name}'"
                            other['error'] = (other.get('error', '') + "----||----\n" if other.get('error') else "") + other_msg
                            print(f"对应type组件名称 -> '{new_name}'")
        return components
    
    """
    处理二级关键字
    """

    def split_sub_blocks(self, comp):
        sub_keywords = ['features', 'subcomponents', 'connections', 'properties', 'calls']
        text = comp['content']
        blocks = []

        pattern = re.compile(r'^\s*(' + '|'.join(sub_keywords) + r')\b', re.IGNORECASE | re.MULTILINE)
        matches = list(pattern.finditer(text))
        end_pattern = re.compile(r'end\b', re.IGNORECASE)

        for i, match in enumerate(matches):
            block_name = match.group(1).lower()
            start_idx = match.start()

            if i + 1 < len(matches):
                end_idx = matches[i + 1].start()
            else:
                end_match = end_pattern.search(text, start_idx)
                end_idx = end_match.start() if end_match else len(text)

            block_content = text[start_idx:end_idx].strip()
            blocks.append({
                'block_name': block_name,
                'content': block_content
            })

        comp['sub_blocks'] = blocks
        print(blocks)
        
        return comp

    def merge_components_to_text(self, components):
        """
        将修复后的所有组件重新拼接成完整的 AADL 文本
        """
        aadl_texts = []
        for comp in components:
            raw = comp.get('raw_text', '').strip()
            if raw:
                aadl_texts.append(raw)
        return '\n\n'.join(aadl_texts)

if __name__ == "__main__":
    aadl_text = """
    system MySystem
    features
        f1: in data port;
    end Mysystem;

    implementation MySystem.impl

    end MySystem;

    process BrokenProcess
    features
        p1: in data port;
        
    thread NamelessEnd
    features
        t1: in data port;
    end;

    device WrongNameDevice
    features
        d1: in data port;
    end SomeDevice;
    
    thread implementation name
    end name;

    bus bus
    end bus;

    bus implementation bus
    end bus;

    thread implementation test.impl
    subcomponents
        name: test
    end name.impl;
    """
    parser = AADLParser(aadl_text)
    components = parser.split_components(aadl_text)
    # for c in components:
        # print(c)
    new01 = parser.fix_implementation_consistency(components)
    out1 = parser.merge_components_to_text(new01)
    print("----------")
    print(out1)
    print("----------")
    new02 = parser.fix_keyword_name_conflicts_name_only(new01)
    output = parser.merge_components_to_text(new02)
    print(output)
    print("----------")
    for c in parser.components:
        c = parser.split_sub_blocks(c)

    # print(parse.extract_aadl_base_info())
    # print(parse.remove_aadl_comments())
