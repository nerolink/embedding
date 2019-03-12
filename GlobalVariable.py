import javalang.tree as jlt
import torch

types = [jlt.FormalParameter, jlt.BasicType, jlt.PackageDeclaration, jlt.InterfaceDeclaration, jlt.CatchClauseParameter,
         jlt.ClassDeclaration,
         jlt.MemberReference, jlt.SuperMemberReference, jlt.ConstructorDeclaration, jlt.ReferenceType,
         jlt.MethodDeclaration, jlt.VariableDeclarator, jlt.IfStatement, jlt.WhileStatement, jlt.DoStatement,
         jlt.ForStatement, jlt.AssertStatement, jlt.BreakStatement, jlt.ContinueStatement, jlt.ReturnStatement,
         jlt.ThrowStatement, jlt.SynchronizedStatement, jlt.TryStatement,
         jlt.SwitchStatement, jlt.BlockStatement, jlt.StatementExpression, jlt.TryResource, jlt.CatchClause,
         jlt.CatchClauseParameter, jlt.SwitchStatementCase, jlt.ForControl, jlt.EnhancedForControl]
word_to_vec = {}
word_to_node = {}
d_type = torch.float64
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
params = {'vec_size': 100, 'echo_size': 1, 'learning_rate': 0.0001}
# projects_source_dir = "C:\\Users\\nero\\Desktop\\papers\\软件测试\\Data\\projects\\"
projects_source_dir = "R:\\"
hf_root = None
training_data = []
debug_map = {}
count = 0
isDebug = True


def print_debug_map():
    with open("debug.txt", 'w') as file_obj:
        for k, v in debug_map.items():
            file_obj.write(str(k) + ":" + str(v))
        file_obj.flush()
