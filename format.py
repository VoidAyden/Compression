import heapq
from collections import defaultdict

# LZ77 Compression Algorithm
def lz77_compress(input_data):
    window_size = 256
    search_buffer = []
    result = []
    i = 0
    while i < len(input_data):
        match = None
        for j in range(max(i - window_size, 0), i):
            k = 0
            while i + k < len(input_data) and input_data[j + k] == input_data[i + k]:
                k += 1
            if k > 1:  # Find matches of length greater than 1
                match = (j, k)
                break

        if match:
            result.append((match[0], match[1], input_data[i + match[1]]))
            i += match[1] + 1  # Skip the matched portion
        else:
            result.append((0, 0, input_data[i]))
            i += 1
    return result

# Huffman Encoding Algorithm
def build_huffman_tree(data):
    freq = defaultdict(int)
    for char in data:
        freq[char] += 1

    # Create a priority queue from the frequency table
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # The Huffman codes are stored in the first element of the heap
    huff_tree = heap[0]
    return dict(sorted([(char, code) for char, code in huff_tree[1:]], key=lambda x: x[1]))

def huffman_encode(data, huff_tree):
    return ''.join([huff_tree[char] for char in data])

def huffman_decode(encoded_data, huff_tree):
    reverse_tree = {code: char for char, code in huff_tree.items()}
    decoded_data = []
    temp_code = ""
    for bit in encoded_data:
        temp_code += bit
        if temp_code in reverse_tree:
            decoded_data.append(reverse_tree[temp_code])
            temp_code = ""
    return ''.join(decoded_data)

# Compression with LZ77 and Huffman Encoding
def compress_directory_data(input_data):
    # Step 1: LZ77 compression
    lz77_compressed_data = lz77_compress(input_data)

    # Convert the LZ77 output (which is a tuple) to a string for Huffman encoding
    lz77_output = ''.join([f"{offset},{length},{next_char}" for offset, length, next_char in lz77_compressed_data])

    # Step 2: Huffman encoding
    huff_tree = build_huffman_tree(lz77_output)
    huffman_compressed_data = huffman_encode(lz77_output, huff_tree)

    return huffman_compressed_data, huff_tree

# Save the custom compressed file with a header
def save_custom_compressed_file(filename, compressed_data, huff_tree):
    with open(filename, 'wb') as f:
        # Write header
        f.write(b"MYCOMPRESSED\n")
        
        # Serialize the Huffman tree (for decoding later)
        f.write(f"{len(huff_tree)}\n".encode())
        for char, code in huff_tree.items():
            f.write(f"{char}:{code}\n".encode())
        
        # Write compressed data
        f.write(compressed_data.encode())

# Read the custom compressed file and return the compressed data and Huffman tree
def read_custom_compressed_file(filename):
    with open(filename, 'rb') as f:
        header = f.readline().decode().strip()
        if header != "MYCOMPRESSED":
            raise ValueError("Invalid file format")

        # Read Huffman tree
        tree_size = int(f.readline().decode().strip())
        huff_tree = {}
        for _ in range(tree_size):
            line = f.readline().decode().strip()
            char, code = line.split(":")
            huff_tree[char] = code
        
        # Read compressed data
        compressed_data = f.read().decode()

    return compressed_data, huff_tree

# Decompress data by reversing Huffman and LZ77 steps
def decompress_data(compressed_data, huff_tree):
    # Step 1: Huffman decode
    decompressed_lz77 = huffman_decode(compressed_data, huff_tree)

    # Step 2: LZ77 decompress (function needs to be defined)
    decompressed_data = lz77_decompress(decompressed_lz77)
    
    return decompressed_data

# LZ77 Decompression (opposite of LZ77 compression)
def lz77_decompress(compressed_data):
    decompressed = []
    for offset, length, next_char in compressed_data:
        if offset == 0 and length == 0:
            decompressed.append(next_char)
        else:
            start = len(decompressed) - offset
            decompressed.extend(decompressed[start:start + length])
            decompressed.append(next_char)
    return ''.join(decompressed)

# Example usage
input_data = "this is a simple example input text for compression"*100  # Example large text input
compressed_data, huff_tree = compress_directory_data(input_data)

# Save to file
save_custom_compressed_file("compressed_data.mycomp", compressed_data, huff_tree)

# Read and decompress from file
compressed_data, huff_tree = read_custom_compressed_file("compressed_data.mycomp")
decompressed_data = decompress_data(compressed_data, huff_tree)

print(f"Decompressed data: {decompressed_data[:100]}...")  # Print first 100 chars to check
