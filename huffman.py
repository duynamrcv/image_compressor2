import cv2
import numpy as np
 
class node:
	# Define node construction method
	def __init__(self, right=None, left=None, parent=None, weight=0, code=None):
		self.left = left 
		self.right = right 
		self.parent = parent 
		self.weight = weight # weight
		self.code = code # Node value
	
# Define function, count the number of times each pixel appears
def pixel_number_caculate(list):
	pixel_number={}
	for i in list:
		if i not in pixel_number.keys():
			pixel_number[i] = 1 # If the pixel is not in the character frequency dictionary, add it directly
		else:
			pixel_number[i] += 1 # If it exists in the character frequency string dictionary, add one to the corresponding value 
	return pixel_number

# Construct a node, give its value and corresponding weight respectively 
def node_construct(pixel_number): 
	node_list =[]
	for i in range(len(pixel_number)):
		node_list.append(node(weight=pixel_number[i][1],code=str(pixel_number[i][0])))
	return node_list

# Construct a node, give its value and corresponding weight respectively 
def node_construct(pixel_number): 
	node_list =[]
	for i in range(len(pixel_number)):
		node_list.append(node(weight=pixel_number[i][1],code=str(pixel_number[i][0])))
	return node_list

# According to the list of leaf nodes, generate the corresponding Huffman coding tree
def tree_construct(listnode):
	listnode = sorted(listnode, key=lambda node:node.weight) 
	while len(listnode) != 1:
		# Each time the two pixel points of the weighted value are merged
		low_node0,low_node1 = listnode[0], listnode[1]
		new_change_node = node()
		new_change_node.weight = low_node0.weight + low_node1.weight
		new_change_node.left = low_node0
		new_change_node.right = low_node1
		low_node0.parent = new_change_node
		low_node1.parent = new_change_node
		listnode.remove(low_node0)
		listnode.remove(low_node1)
		listnode.append(new_change_node)
		listnode = sorted(listnode, key=lambda node:node.weight) 
	return listnode

def encoding(img):
	img = img.astype(np.int16)
	rows, cols = img.shape[:2]

	list =[]
	for i in range(rows):
		for j in range(cols): 
			list.append(img[i,j])
	# Count the number of times of each pixel, and sort according to the number of occurrences from small to large 
	pixel_number = pixel_number_caculate(list)
	pixel_number = sorted(pixel_number.items(),key=lambda item:item[1])
		
	# Construct the node list according to the value of the pixel and the number of occurrences 
	node_list = node_construct(pixel_number)
	# Construct a Huffman tree, save the head node
	head = tree_construct(node_list)[0]
		
	# Construction code table
	coding_table = {}
	for e in node_list:
		new_change_node = e
		coding_table.setdefault(e.code,"")
		while new_change_node !=head:
			if new_change_node.parent.left == new_change_node: 
				coding_table[e.code] = "1" + coding_table[e.code] 
			else:
				coding_table[e.code] = "0" + coding_table[e.code] 
			new_change_node = new_change_node. parent

	# Convert the encoding result of the image into a string and save it in txt 
	coding_result = ''
	for i in range(rows):
		for j in range(cols):
			for key,values in coding_table.items(): 
				if str(img[i,j]) == key:
					coding_result = coding_result + values 
	
	# print(coding_table)
	
	print("ENCODE DONE!!!")
	return coding_table, coding_result

def decoding(rows, cols, coding_table, coding_result): 
	code_read_now=''#The currently read code 
	new_pixel =[] 
	i = 0
	while (i != coding_result.__len__()):
		# Read one later each time
		code_read_now = code_read_now + coding_result[i]
		for key in coding_table.keys():
			# If the currently read code exists in the code table 
			if code_read_now == coding_table[key]: 
				new_pixel.append(key) 
				code_read_now = ''
				break
		i += 1
	
	# Construct a new image
	decode_image = np.zeros((rows, cols), dtype=np.int16)
	k = 0

	for i in range(rows):
		for j in range(cols):
			decode_image[i,j] = int(new_pixel[k])
			k+=1
	print("DECODE DONE!!!")
	return decode_image

# img = cv2.imread("images/lena.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# coding_table, coding_result = encoding(gray)

# rows, cols = img.shape[:2]
# decode_image = decoding(rows, cols, coding_table, coding_result)