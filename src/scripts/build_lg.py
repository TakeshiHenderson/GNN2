import sys
import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm

class InkmlToLG:
    def __init__(self):
        pass

    def _get_tag(self, element):
        """
        Extracts tag name ignoring namespaces.
        Example: '{http://www.w3.org/1998/Math/MathML}mfrac' -> 'mfrac'
        """
        return element.tag.split('}')[-1]

    def _get_xml_id(self, element):
        """
        Aggressively searches for any attribute that looks like an ID.
        Handles 'xml:id', 'id', and namespaced variants.
        """
        if 'id' in element.attrib:
            return element.attrib['id']
            
        # Iterate over all attributes to find one ending in 'id'
        # This handles '{http://www.w3.org/XML/1998/namespace}id' automatically
        for k, v in element.attrib.items():
            if k.endswith('id'): 
                return v
        return None

    def convert(self, inkml_path, output_path):
        try:
            tree = ET.parse(inkml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[Error] Failed to parse {inkml_path}: {e}")
            return

        # --- 1. Extract Objects (TraceGroups) ---
        objects = []
        xml_id_to_obj_id = {}
        
        # Find all traceGroups (Namespace Agnostic)
        all_elements = list(root.iter())
        trace_groups = [e for e in all_elements if self._get_tag(e) == 'traceGroup']
        
        obj_counter = 0
        
        for tg in trace_groups:
            # 1. Get Label (annotation type="truth")
            label = "?"
            for child in tg:
                if self._get_tag(child) == 'annotation' and child.get('type') == 'truth':
                    label = child.text
                    break
            
            # 2. Get Strokes (traceView)
            strokes = []
            for child in tg:
                if self._get_tag(child) == 'traceView':
                    ref = child.get('traceDataRef')
                    if ref: strokes.append(ref)
            
            # If no strokes, skip (unless it's a special container, but usually symbols have strokes)
            if not strokes:
                continue

            # 3. Get XML ID Link (annotationXML href=...)
            xml_id = None
            for child in tg:
                if self._get_tag(child) == 'annotationXML':
                    xml_id = child.get('href')
                    break
            
            # Generate Internal ID
            obj_id = f"O_{obj_counter}"
            obj_counter += 1
            
            # Store Mapping
            if xml_id:
                xml_id_to_obj_id[xml_id] = obj_id
            
            # Store Object Line
            stroke_str = ", ".join(strokes)
            objects.append(f"O, {obj_id}, {label}, 1.0, {stroke_str}")

        # --- 2. Extract Relations (MathML) ---
        relations = []
        
        # Find <math> tag anywhere in the file
        math_elem = None
        for elem in all_elements:
            if self._get_tag(elem) == 'math':
                math_elem = elem
                break
        
        if math_elem is not None:
            self._parse_mathml_recursive(math_elem, xml_id_to_obj_id, relations)
        else:
            print(f"[Warning] No <math> tag found in {inkml_path}")

        # --- 3. Write Output ---
        with open(output_path, 'w') as f:
            f.write(f"# Objects ({len(objects)}):\n")
            for line in objects:
                f.write(line + "\n")
            f.write(f"\n# Relations ({len(relations)}):\n")
            for line in relations:
                f.write(line + "\n")

    def _parse_mathml_recursive(self, element, xml_map, relations):
        """
        Traverses MathML tree to generate EO lines.
        Returns: The Object ID that represents the 'Root' of this subgraph.
        """
        tag = self._get_tag(element)
        xml_id = self._get_xml_id(element)
        
        # Case A: This element IS a Symbol (Leaf)
        # If this tag has an ID that maps to a TraceGroup, it is a Node.
        if xml_id and xml_id in xml_map:
            my_obj_id = xml_map[xml_id]
            
            # However, if it is a container like mfrac or msqrt, we still need to process children
            # to link them TO this node.
            
            # Process Children
            child_roots = []
            for child in element:
                c_id = self._parse_mathml_recursive(child, xml_map, relations)
                child_roots.append(c_id)
            
            valid_children = [c for c in child_roots if c is not None]

            # 1. Fraction (mfrac) linked to Bar
            if tag == 'mfrac':
                # If we have at least one child, we can link it
                if len(valid_children) >= 1:
                    relations.append(f"EO, {my_obj_id}, {valid_children[0]}, Above, 1.0")
                if len(valid_children) >= 2:
                    relations.append(f"EO, {my_obj_id}, {valid_children[1]}, Below, 1.0")
            
            # 2. Radical (msqrt) linked to Radical Symbol
            elif tag == 'msqrt':
                # Link all valid children (usually just one block) as Inside
                if valid_children:
                    relations.append(f"EO, {my_obj_id}, {valid_children[0]}, Inside, 1.0")
            
            return my_obj_id

        # Case B: Structural Container (No direct Object ID)
        # e.g., mrow, math, mstyle
        else:
            # Process Children
            child_roots = []
            for child in element:
                c_id = self._parse_mathml_recursive(child, xml_map, relations)
                child_roots.append(c_id)
            
            valid_children = [c for c in child_roots if c is not None]
            
            # 1. Implicit Row (Right Relations)
            # Tags: math, mrow, mstyle, mpadded, phantom, semantics
            if tag in ['math', 'mrow', 'mstyle', 'mpadded', 'mphantom', 'semantics', 'annotationXML']:
                for i in range(len(valid_children) - 1):
                    src, dst = valid_children[i], valid_children[i+1]
                    relations.append(f"EO, {src}, {dst}, Right, 1.0")
                
                return valid_children[0] if valid_children else None

            # 2. Subscript (msub)
            elif tag == 'msub':
                if len(valid_children) >= 2:
                    relations.append(f"EO, {valid_children[0]}, {valid_children[1]}, Sub, 1.0")
                return valid_children[0] if valid_children else None

            # 3. Superscript (msup)
            elif tag == 'msup':
                if len(valid_children) >= 2:
                    relations.append(f"EO, {valid_children[0]}, {valid_children[1]}, Sup, 1.0")
                return valid_children[0] if valid_children else None

            # 4. SubSuperscript (msubsup)
            elif tag == 'msubsup':
                if len(valid_children) >= 3:
                    base, sub, sup = valid_children[0], valid_children[1], valid_children[2]
                    relations.append(f"EO, {base}, {sub}, Sub, 1.0")
                    relations.append(f"EO, {base}, {sup}, Sup, 1.0")
                return valid_children[0] if valid_children else None
            
            # 5. Underscript/Overscript (munder, mover) -> Treat as Below/Above or Sub/Sup
            elif tag == 'munder':
                if len(valid_children) >= 2:
                    relations.append(f"EO, {valid_children[0]}, {valid_children[1]}, Below, 1.0")
                return valid_children[0] if valid_children else None
            
            elif tag == 'mover':
                if len(valid_children) >= 2:
                    relations.append(f"EO, {valid_children[0]}, {valid_children[1]}, Above, 1.0")
                return valid_children[0] if valid_children else None

            # Default: Pass through the first child
            return valid_children[0] if valid_children else None


if __name__ == "__main__":
    converter = InkmlToLG()
    
    # Test on your specific file
    # Ensure this points to the correct location
    # converter.convert("/home/takeshi/Documents/AOL DL/crohme_dataset/train/inkml/MfrDB1288.inkml", "/home/takeshi/Documents/AOL DL/crohme_dataset/train/lg_new_1/MfrDB1288.lg")
    # print("Conversion complete. Check MfrDB1288.lg")
    input_path = "/home/takeshi/Documents/AOL DL/crohme_dataset/test/inkml_gt/0046.inkml"
    output_path = "./temp.lg"
    converter.convert(input_path, output_path)
    print(f"Output written to: {output_path}")
    with open(output_path, "r") as f:
        print(f.read())


# if __name__ == "__main__":
#     converter = InkmlToLG()
    
#     # Example Usage: Process a directory
#     input_dir = "../crohme_dataset/test/inkml_gt/"
#     output_dir = "../crohme_dataset/test/lg_new_1/"
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     files = glob.glob(os.path.join(input_dir, "*.inkml"))
#     print(f"Found {len(files)} InkML files.")
    
#     for f_path in tqdm(files):
#         fname = os.path.basename(f_path)
#         name_only = os.path.splitext(fname)[0]
#         out_path = os.path.join(output_dir, name_only + ".lg")
        
#         converter.convert(f_path, out_path)