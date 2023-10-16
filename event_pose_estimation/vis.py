
def vis(batch_vertices, faces):
    batch_vertices.reshape(-1, 6890, 3)

    for i in range(batch_vertices.shape[0]):
        verts = batch_vertices[i]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        filename = f"{datetime.datetime.now()}.obj"
        mesh.export(f"/home/rowan/source/HPE/EventHPE/data_event/vis/{filename}")
        print(f"Saved mesh to {filename}")

