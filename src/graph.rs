use std::collections::{HashSet, HashMap};
use tabbycat::{GraphBuilder, GraphType, Identity, Edge, StmtList, AttrList, AttrType};
use tabbycat::attributes::{shape, label, Shape, RankDir, rankdir};
use crate::{Value, Operation};

fn trace(root: &Value) -> (HashMap<usize, Value>, HashSet<(usize, usize)>) {
    let mut nodes = HashMap::new();
    let mut edges = HashSet::new();

    fn build(
        parent: &Value,
        nodes: &mut HashMap<usize, Value>,
        edges: &mut HashSet<(usize, usize)>,
    ) {
        if !nodes.contains_key(&parent.id()) {
            nodes.insert(parent.id(), parent.clone());
            for child in parent.previous() {
                edges.insert((child.id(), parent.id()));
                build(&child, nodes, edges);
            }
        }
    }

    build(&root, &mut nodes, &mut edges);

    (nodes, edges)
}

pub fn create_graph(root: &Value) -> tabbycat::Graph {
    let (nodes, edges) = trace(&root);

    GraphBuilder::default()
        .strict(true)
        .graph_type(GraphType::DiGraph)
        .id(Identity::id("G").unwrap())
        .stmts(
            StmtList::new()
                .add_attr(
                    AttrType::Graph,
                    AttrList::new()
                        .add_pair(rankdir(RankDir::LR))
                )
                .add_attr(
                    AttrType::Node,
                    AttrList::new()
                        .add_pair(
                            // Tabbycat doesn't seem to support: `shape(Shape::Mrecord)`
                            (Identity::id("shape").unwrap(), Identity::id("Mrecord").unwrap())
                        )
                ).extend(
                    // Node statements
                    nodes.iter().map(|(node_id, node)| {
                        let node_identity = Identity::id(format!("node_{}", node_id)).unwrap();
                
                        StmtList::new()
                            .add_node(
                                node_identity.clone(),
                                None,
                                Some(
                                    AttrList::new()
                                        .add_pair(
                                            label(
                                                &format!(
                                                    "{} | data: {:.2} | gradient: {:.2}",
                                                    node.name().unwrap_or("?".to_string()),
                                                    node.data(),
                                                    node.gradient()
                                                )
                                            )
                                        )
                                )
                            ).extend(
                            if node.operation() != Operation::Input {
                                let node_operation_identity = Identity::id(format!("node_{}_{:?}", node_id, node.operation())).unwrap();
                
                                StmtList::new()
                                    .add_node(
                                        node_operation_identity.clone(),
                                        None,
                                        Some(
                                            AttrList::new()
                                                .add_pair(label(&format!("{:?}", node.operation())))
                                                .add_pair(shape(Shape::Circle))
                                            )
                                    ).add_edge(
                                        Edge::head_node(node_operation_identity.clone(), None)
                                            .arrow_to_node(node_identity.clone(), None)
                                    )
                            } else { StmtList::new() }
                        )
                    }).fold(StmtList::new(), |acc, statements| { acc.extend(statements) })
                )
                .extend(
                    // Edge statements
                    edges.iter().map(|(child_id, parent_id)| {
                        let parent = nodes.get(parent_id).unwrap();
                
                        let child_node_identity = Identity::id(format!("node_{}", child_id)).unwrap();
                        let parent_node_operation_identity = Identity::id(format!("node_{}_{:?}", parent_id, parent.operation())).unwrap();
                
                        StmtList::new().add_edge(
                            Edge::head_node(child_node_identity.clone(), None)
                                .arrow_to_node(parent_node_operation_identity.clone(), None)
                        )
                    }).fold(StmtList::new(), |acc, statements| { acc.extend(statements) })
                )
        )
        .build()
        .unwrap()
}
